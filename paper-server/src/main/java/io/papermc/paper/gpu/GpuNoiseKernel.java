package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

public final class GpuNoiseKernel {

    private static final Logger LOGGER = Logger.getLogger("GlassPaper");
    // Must match GpuContext.DENSITY_QUEUE_COUNT — each slot pairs 1:1 with one queue.
    private static final int POOL_SIZE = GpuContext.DENSITY_QUEUE_COUNT;

    // Phase 9.4 — buffer pooling for the hot density-tree dispatch path.
    // Each slot pre-allocates posBuf (input positions) and outBuf (results)
    // sized for the largest natural batch we expect from GpuDispatchQueue:
    // ~8 workers × ~5 interpolators × ~825 points ≈ 33K positions per group.
    // Dispatches above this cap fall back to alloc-on-demand so we never
    // crash on outliers — they just pay the old per-call overhead.
    //
    // Removes ~20-50µs of cl_mem create/release latency per dispatch.
    private static final int MAX_POOLED_POINTS = 32768;

    // Phase 9.13 — per-slot ring depth for async dispatch. Each slot keeps
    // RING_DEPTH worth of (posBuf, outBuf, hostResults, kernel) tuples. The
    // dispatch thread can enqueue a new write/kernel/read in any free ring
    // entry without waiting for prior dispatches on the same slot to
    // complete. The slot's command queue serializes them in submission
    // order (in-order semantics), but the host doesn't block per-dispatch.
    //
    // A dedicated completion thread waits on the read event of each in-flight
    // ring entry and runs the user callback (split merged result back into
    // per-item slices, complete the worker futures). This decouples
    // worker-future delivery from the dispatch-thread's enqueue path.
    //
    // K=4 was chosen as a balance:
    //   - Higher K = more concurrent device work, more memory.
    //   - K=4 with 8 slots = up to 32 in-flight, matches NVIDIA Hyper-Q's
    //     max concurrent streams on Turing (RTX 2080 Ti).
    //   - Memory: 8 slots × 4 ring × (786 KB pos + 262 KB out) ≈ 33 MB GPU
    //     plus 8 × 4 × 32 KB ≈ 1 MB host. Negligible.
    private static final int RING_DEPTH = 4;

    /**
     * Phase 9.13 — callback invoked by the completion thread when a previously
     * enqueued async dispatch's read DMA has finished. Receives the
     * host-side result buffer and the point count that was enqueued.
     *
     * Runs on the completion thread, NOT on the dispatch pool thread that
     * enqueued the work. Should be quick (e.g. arraycopy + future.complete);
     * any heavy work should be deferred to the consumer thread.
     *
     * If the dispatch fails (e.g. CL error mid-enqueue), the callback is
     * invoked with onFailure() instead. The implementation is responsible
     * for signaling CPU fallback to its consumers.
     */
    public interface AsyncCallback {
        void onComplete(double[] hostResults, int pointCount);
        default void onFailure(Throwable cause) {}
    }

    /**
     * Phase 9.13 — one slot of the per-slot ring. Each ring entry owns its
     * own (posBuf, outBuf, hostResults, kernel) so multiple dispatches can
     * be enqueued on the same slot's command queue without their buffers
     * stomping on each other.
     *
     * The kernel object is per-ring-entry (not shared across the slot) so
     * Phase 9.12.A's static arg state survives across cycles of this ring
     * entry. Arg 0 (posBuf) and arg 13 (outBuf) are bound once at init.
     * lastGpuKernel tracks args 1-12.
     *
     * State machine:
     *   FREE: entry not in use; safe for a dispatch thread to enqueue
     *   IN_FLIGHT: enqueueWrite/Kernel/Read have been called; completion
     *              thread will eventually move it back to FREE
     */
    private static final class RingEntry {
        final cl_kernel kernel;
        final cl_mem    posBuf;
        final cl_mem    outBuf;
        final double[]  hostResults;  // capacity MAX_POOLED_POINTS
        GpuCompiledKernel lastGpuKernel;  // Phase 9.12.A — per-ring-entry arg cache

        // In-flight state (mutated by dispatch thread under slot borrow,
        // read by completion thread). The handoff is via the completion
        // queue — once a (ringEntry, event, callback) tuple is pushed,
        // the dispatch thread must not touch the entry until the completion
        // thread signals it free via processed.complete().
        volatile cl_event readEvent;
        volatile int      pointCount;
        volatile AsyncCallback callback;
        volatile double[] positionsRef;  // GC root for the input array until DMA completes

        // CompletableFuture-based handoff: the dispatch thread, when picking
        // a ring entry it just used, waits on this to ensure the completion
        // thread is done with it. The completion thread completes this
        // future after invoking the callback and releasing the cl_event.
        // null means the entry has never been used.
        volatile java.util.concurrent.CompletableFuture<Void> processed;

        RingEntry(cl_kernel k, cl_mem pb, cl_mem ob) {
            this.kernel      = k;
            this.posBuf      = pb;
            this.outBuf      = ob;
            this.hostResults = new double[MAX_POOLED_POINTS];
        }
    }

    /**
     * Density-tree slot — bundles a per-slot in-order command queue (Phase 9.5)
     * with a ring of RING_DEPTH RingEntry tuples (Phase 9.13). Each slot is
     * borrowed by at most one dispatcher thread at a time, so the queue is
     * single-writer and needs no mutex. Different slots' queues run
     * concurrently on the device (NVIDIA: separate CUDA streams; AMD:
     * separate ACE hardware queues, round-robined by creation order).
     *
     * Phase 9.13: instead of one (kernel, posBuf, outBuf) per slot the slot
     * has RING_DEPTH of them in a ring buffer, accessed round-robin via
     * nextRingIndex. The dispatch thread, on borrow, picks ring[i] and
     * waits (if needed) for the completion thread to have finished any
     * previous use of ring[i] before enqueueing on it. With ring depth 4
     * and the dispatch thread + completion thread typically operating at
     * similar rates, the wait is rare in steady state.
     *
     * Kept legacy sync fields (kernel, posBuf, outBuf, lastGpuKernel) for
     * the validator path that calls evalDensityTreeFast at startup — those
     * use blocking writes/reads and don't engage the ring.
     */
    private static final class DensitySlot {
        // Phase 9.5 — per-slot in-order command queue
        final cl_command_queue queue;

        // Phase 9.13 — async dispatch ring
        final RingEntry[] ring;
        int nextRingIndex;  // mutated only by dispatch thread under slot borrow

        // Legacy sync path — used by validators and the unpooled fallback.
        // Owns its own kernel/posBuf/outBuf so it can coexist on the same
        // command queue as ring entries without arg-state collision.
        final cl_kernel kernel;
        final cl_mem    posBuf;
        final cl_mem    outBuf;
        GpuCompiledKernel lastGpuKernel;  // Phase 9.12.A — for the sync path

        DensitySlot(cl_command_queue q, RingEntry[] ring,
                    cl_kernel k, cl_mem pb, cl_mem ob) {
            this.queue   = q;
            this.ring    = ring;
            this.kernel  = k;
            this.posBuf  = pb;
            this.outBuf  = ob;
            this.lastGpuKernel = null;
            this.nextRingIndex = 0;
        }
    }

    /**
     * Phase 9.13 — pending completion record. Pushed by the dispatch thread
     * after enqueueing the async write/kernel/read; popped by the completion
     * thread, which waits on the readEvent and invokes the callback.
     */
    private static final class PendingCompletion {
        final RingEntry entry;
        // Holding refs to the cl_event and the slot's queue is sufficient;
        // everything else lives on the RingEntry.
        PendingCompletion(RingEntry entry) {
            this.entry = entry;
        }
    }

    private final GpuContext ctx;
    private final cl_program program;
    private final BlockingQueue<cl_kernel>    noisePool           = new LinkedBlockingQueue<>();
    private final BlockingQueue<cl_kernel>    normalNoisePool     = new LinkedBlockingQueue<>();
    private final BlockingQueue<DensitySlot>  densityTreeSlotPool = new LinkedBlockingQueue<>();

    // Phase 9.6.F — local work-group size for the density-tree kernel,
    // determined at build time from the device's preferred SIMD width and
    // the kernel's resource budget. Replaces the hardcoded 64 used since
    // Phase 7. Portable: NVIDIA warp = 32, AMD wavefront = 64, Intel SIMD8/16/32.
    private final int densityLocalWorkSize;

    // Phase 9.13 — async dispatch infrastructure.
    // pendingCompletions: dispatch thread pushes (RingEntry) after enqueueing
    // async work; completion thread takes one, clWaitForEvents on its
    // entry.readEvent, invokes entry.callback, releases the event, and
    // completes entry.processed so a future dispatch can reuse this entry.
    //
    // BlockingQueue gives the completion thread efficient wakeup on push
    // and bounded by JVM heap (we never queue more than POOL_SIZE × RING_DEPTH
    // before backpressure kicks in on the dispatch side).
    private final BlockingQueue<PendingCompletion> pendingCompletions = new LinkedBlockingQueue<>();
    private Thread completionThread;
    private volatile boolean completionRunning;

    private GpuNoiseKernel(GpuContext ctx, cl_program program, int densityLocalWorkSize) {
        this.ctx     = ctx;
        this.program = program;
        this.densityLocalWorkSize = densityLocalWorkSize;
    }

    public static GpuNoiseKernel build() {
        GpuContext ctx = GpuContext.get();
        if (ctx == null) {
            LOGGER.warning("GpuNoiseKernel: no GPU context, skipping.");
            return null;
        }

        String source;
        try (InputStream is = GpuNoiseKernel.class.getResourceAsStream("/gpu/density.cl")) {
            if (is == null) { LOGGER.severe("density.cl not found!"); return null; }
            source = new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            LOGGER.severe("Failed to read density.cl: " + e.getMessage());
            return null;
        }

        cl_program program = clCreateProgramWithSource(ctx.context(), 1, new String[]{source}, null, null);
        int buildResult = clBuildProgram(program, 1, new cl_device_id[]{ctx.device()},
            "-cl-mad-enable -cl-fast-relaxed-math", null, null);

        long[] logSize = new long[1];
        clGetProgramBuildInfo(program, ctx.device(), CL_PROGRAM_BUILD_LOG, 0, null, logSize);
        if (logSize[0] > 2) {
            byte[] logBytes = new byte[(int) logSize[0]];
            clGetProgramBuildInfo(program, ctx.device(), CL_PROGRAM_BUILD_LOG,
                logBytes.length, Pointer.to(logBytes), null);
            LOGGER.info("Kernel build log:\n" +
                new String(logBytes, StandardCharsets.UTF_8).trim());
        }

        if (buildResult != CL_SUCCESS) {
            LOGGER.severe("density.cl compilation FAILED (code " + buildResult + ")");
            clReleaseProgram(program);
            return null;
        }

        // Build one density-tree kernel first so we can query its per-device
        // resource budget. CL_KERNEL_WORK_GROUP_SIZE may be smaller than
        // CL_DEVICE_MAX_WORK_GROUP_SIZE if the kernel uses many registers
        // (our 64-double stack does spill on most devices).
        cl_kernel sampleDk = clCreateKernel(program, "evalDensityTree", null);
        int localWorkSize = pickLocalWorkSize(sampleDk, ctx.device());
        clReleaseKernel(sampleDk);

        GpuNoiseKernel result = new GpuNoiseKernel(ctx, program, localWorkSize);
        for (int i = 0; i < POOL_SIZE; i++) {
            result.noisePool.add(clCreateKernel(program, "sampleNoise", null));
            result.normalNoisePool.add(clCreateKernel(program, "sampleNormalNoise", null));

            // Pre-allocate one (kernel, posBuf, outBuf, queue) tuple per
            // density slot. Buffers stay resident for the lifetime of the
            // kernel; positions are re-uploaded with clEnqueueWriteBuffer on
            // each dispatch. The dedicated queue means dispatches from
            // different slots run concurrently on the device.
            // Sync path's kernel + buffers (Phase 9.4 + 9.12.A).
            // Used by validators at startup. Bound once with its own posBuf
            // / outBuf so the sync path doesn't collide with ring entries'
            // arg state on the same slot.
            cl_kernel dk = clCreateKernel(program, "evalDensityTree", null);
            cl_mem pb = clCreateBuffer(ctx.context(),
                CL_MEM_READ_ONLY,
                (long) Sizeof.cl_double * MAX_POOLED_POINTS * 3, null, null);
            cl_mem ob = clCreateBuffer(ctx.context(),
                CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_double * MAX_POOLED_POINTS, null, null);
            cl_command_queue dq = ctx.densityQueue(i);

            // Phase 9.12.A — bind sync-path persistent args once.
            clSetKernelArg(dk,  0, Sizeof.cl_mem, Pointer.to(pb));
            clSetKernelArg(dk, 13, Sizeof.cl_mem, Pointer.to(ob));

            // Phase 9.13 — allocate ring of RING_DEPTH (kernel, posBuf, outBuf)
            // tuples. Each ring entry's kernel has its arg 0 / arg 13 bound
            // once to its own buffers, so the static-arg cache (9.12.A) only
            // needs to manage args 1-12 from the GpuCompiledKernel.
            RingEntry[] ring = new RingEntry[RING_DEPTH];
            for (int r = 0; r < RING_DEPTH; r++) {
                cl_kernel rk = clCreateKernel(program, "evalDensityTree", null);
                cl_mem rpb = clCreateBuffer(ctx.context(),
                    CL_MEM_READ_ONLY,
                    (long) Sizeof.cl_double * MAX_POOLED_POINTS * 3, null, null);
                cl_mem rob = clCreateBuffer(ctx.context(),
                    CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_double * MAX_POOLED_POINTS, null, null);
                clSetKernelArg(rk,  0, Sizeof.cl_mem, Pointer.to(rpb));
                clSetKernelArg(rk, 13, Sizeof.cl_mem, Pointer.to(rob));
                ring[r] = new RingEntry(rk, rpb, rob);
            }

            result.densityTreeSlotPool.add(new DensitySlot(dq, ring, dk, pb, ob));
        }

        // Pooled memory: sync path's buffers + ring entries' buffers.
        long syncBytes = (long) POOL_SIZE
            * (Sizeof.cl_double * MAX_POOLED_POINTS * 3
             + Sizeof.cl_double * MAX_POOLED_POINTS);
        long ringBytes = (long) POOL_SIZE * RING_DEPTH
            * (Sizeof.cl_double * MAX_POOLED_POINTS * 3
             + Sizeof.cl_double * MAX_POOLED_POINTS);
        LOGGER.info(String.format(
            "density.cl compiled, kernel ready. Pre-allocated %d density slots "
          + "× (1 sync + %d async ring entries), %d points cap each, "
          + "%.1f MB pooled GPU memory total. Local work-size: %d.",
            POOL_SIZE, RING_DEPTH, MAX_POOLED_POINTS,
            (syncBytes + ringBytes) / (1024.0 * 1024.0),
            localWorkSize));

        // Phase 9.13 — start the completion thread that drains
        // pendingCompletions, waits on each entry's read event, invokes
        // its callback, and frees the ring entry for reuse.
        result.startCompletionThread();

        return result;
    }

    /**
     * Phase 9.13 — completion thread. Drains the pendingCompletions queue,
     * one entry at a time, and for each:
     *   1. clWaitForEvents on entry.readEvent — blocks until the device
     *      finishes the kernel + DMA read on this ring entry.
     *   2. Invoke entry.callback with the host-side result buffer.
     *   3. clReleaseEvent and clear in-flight state.
     *   4. Complete entry.processed so a future dispatch on this ring entry
     *      can proceed.
     *
     * Single thread is sufficient: it's I/O-bound (blocks in
     * clWaitForEvents) so it only consumes CPU when work completes; events
     * complete roughly at the device's throughput rate (~one every 200 µs
     * at 5× overlap and 1 ms wall), so a single thread keeps up easily.
     *
     * If we ever measure thread-bound completion, we can shard by slot.
     */
    private void startCompletionThread() {
        completionRunning = true;
        completionThread = new Thread(() -> {
            while (completionRunning) {
                PendingCompletion pc;
                try {
                    pc = pendingCompletions.poll(50, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
                if (pc == null) continue;

                RingEntry entry = pc.entry;
                cl_event ev = entry.readEvent;
                AsyncCallback cb = entry.callback;
                double[] results = entry.hostResults;
                int pointCount = entry.pointCount;

                try {
                    if (ev != null) {
                        clWaitForEvents(1, new cl_event[]{ev});
                    }
                    if (cb != null) {
                        try {
                            cb.onComplete(results, pointCount);
                        } catch (Throwable t) {
                            LOGGER.warning("GPU completion callback failed: " + t.getMessage());
                        }
                    }
                } catch (Throwable t) {
                    LOGGER.warning("GPU completion failed: " + t.getMessage());
                    if (cb != null) {
                        try { cb.onFailure(t); } catch (Throwable ignored) {}
                    }
                } finally {
                    if (ev != null) {
                        try { clReleaseEvent(ev); } catch (Throwable ignored) {}
                    }
                    // Clear in-flight state. The dispatch thread that next
                    // picks up this ring entry will see processed==done and
                    // proceed without waiting.
                    entry.readEvent    = null;
                    entry.callback     = null;
                    entry.positionsRef = null;
                    // Complete the handoff future; any dispatch thread
                    // waiting on it now unblocks.
                    java.util.concurrent.CompletableFuture<Void> f = entry.processed;
                    if (f != null) f.complete(null);
                }
            }
        }, "GlassPaper-GPU-Completion");
        completionThread.setDaemon(true);
        completionThread.setPriority(Thread.NORM_PRIORITY + 1);
        completionThread.start();
    }

    /**
     * Phase 9.6.F — pick a local work-size that aligns to the device's SIMD
     * width while staying within the kernel's per-WG resource budget.
     *
     * Strategy (portable across NVIDIA/AMD/Intel/CPU OpenCL):
     *   1. Cap at 64 — empirically known good for the density-tree kernel.
     *      Larger work-groups increase register/private-memory pressure
     *      from the 64-double stack and tend to drop occupancy.
     *   2. Cap further at CL_KERNEL_WORK_GROUP_SIZE for this kernel on this
     *      device (smaller than DEVICE_MAX_WORK_GROUP_SIZE if registers spill).
     *   3. Round down to a multiple of CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_
     *      MULTIPLE so we never have a partial SIMD lane.
     *   4. If rounding gives 0 (preferred > cap), fall back to preferred.
     */
    private static int pickLocalWorkSize(cl_kernel kernel, cl_device_id device) {
        long[] tmp = new long[1];
        clGetKernelWorkGroupInfo(kernel, device,
            CL_KERNEL_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(tmp), null);
        long maxKernel = tmp[0];
        clGetKernelWorkGroupInfo(kernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, Sizeof.size_t,
            Pointer.to(tmp), null);
        long preferred = tmp[0];
        if (preferred <= 0) preferred = 1;  // shouldn't happen, but guard

        long target = Math.min(64L, maxKernel);
        long aligned = (target / preferred) * preferred;
        if (aligned <= 0) aligned = Math.min(preferred, maxKernel);
        int picked = (int) Math.max(1, aligned);
        LOGGER.info(String.format(
            "Kernel work-group budget: max=%d, preferred multiple=%d -> picked %d",
            maxKernel, preferred, picked));
        return picked;
    }

    // ── Borrow / return helpers ───────────────────────────────────────────────

    private cl_kernel borrow(BlockingQueue<cl_kernel> pool) {
        try {
            return pool.take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("GPU kernel pool interrupted", e);
        }
    }

    private void ret(BlockingQueue<cl_kernel> pool, cl_kernel k) {
        pool.add(k);
    }

    private DensitySlot borrowSlot() {
        try {
            return densityTreeSlotPool.take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("GPU density slot pool interrupted", e);
        }
    }

    private void returnSlot(DensitySlot s) {
        densityTreeSlotPool.add(s);
    }

    // ── sampleBatch (single ImprovedNoise, used for validation) ──────────────

    public double[] sampleBatch(double[] positions, byte[] permTable,
                                double xo, double yo, double zo, int count) {
        cl_kernel k = borrow(noisePool);
        try {
            cl_mem posBuf  = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * positions.length, Pointer.to(positions), null);
            cl_mem permBuf = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                256L, Pointer.to(permTable), null);
            cl_mem outBuf  = clCreateBuffer(ctx.context(), CL_MEM_WRITE_ONLY,
                (long)Sizeof.cl_double * count, null, null);

            clSetKernelArg(k, 0, Sizeof.cl_mem,    Pointer.to(posBuf));
            clSetKernelArg(k, 1, Sizeof.cl_mem,    Pointer.to(permBuf));
            clSetKernelArg(k, 2, Sizeof.cl_double, Pointer.to(new double[]{xo}));
            clSetKernelArg(k, 3, Sizeof.cl_double, Pointer.to(new double[]{yo}));
            clSetKernelArg(k, 4, Sizeof.cl_double, Pointer.to(new double[]{zo}));
            clSetKernelArg(k, 5, Sizeof.cl_mem,    Pointer.to(outBuf));
            clSetKernelArg(k, 6, Sizeof.cl_int,    Pointer.to(new int[]{count}));

            double[] results = new double[count];
            synchronized (ctx.queue()) {
                clEnqueueNDRangeKernel(ctx.queue(), k, 1, null,
                    new long[]{roundUp(count, 64)}, new long[]{64}, 0, null, null);
                clEnqueueReadBuffer(ctx.queue(), outBuf, CL_TRUE, 0,
                    (long)Sizeof.cl_double * count, Pointer.to(results), 0, null, null);
            }

            clReleaseMemObject(posBuf);
            clReleaseMemObject(permBuf);
            clReleaseMemObject(outBuf);
            return results;
        } finally {
            ret(noisePool, k);
        }
    }

    // ── sampleNormalNoiseBatch (used for validation) ──────────────────────────

    public double[] sampleNormalNoiseBatch(double[] positions, NormalNoiseGpuData data, int count) {
        cl_kernel k = borrow(normalNoisePool);
        try {
            cl_mem posBuf    = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * positions.length, Pointer.to(positions), null);
            cl_mem nParamBuf = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * data.noiseParams.length, Pointer.to(data.noiseParams), null);
            cl_mem octBuf    = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * data.octaveParams.length, Pointer.to(data.octaveParams), null);
            cl_mem permBuf   = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                data.permTables.length, Pointer.to(data.permTables), null);
            cl_mem outBuf    = clCreateBuffer(ctx.context(), CL_MEM_WRITE_ONLY,
                (long)Sizeof.cl_double * count, null, null);

            clSetKernelArg(k, 0, Sizeof.cl_mem, Pointer.to(posBuf));
            clSetKernelArg(k, 1, Sizeof.cl_mem, Pointer.to(nParamBuf));
            clSetKernelArg(k, 2, Sizeof.cl_mem, Pointer.to(octBuf));
            clSetKernelArg(k, 3, Sizeof.cl_mem, Pointer.to(permBuf));
            clSetKernelArg(k, 4, Sizeof.cl_mem, Pointer.to(outBuf));
            clSetKernelArg(k, 5, Sizeof.cl_int, Pointer.to(new int[]{data.firstNumOctaves}));
            clSetKernelArg(k, 6, Sizeof.cl_int, Pointer.to(new int[]{data.secondNumOctaves}));
            clSetKernelArg(k, 7, Sizeof.cl_int, Pointer.to(new int[]{count}));

            double[] results = new double[count];
            synchronized (ctx.queue()) {
                clEnqueueNDRangeKernel(ctx.queue(), k, 1, null,
                    new long[]{roundUp(count, 64)}, new long[]{64}, 0, null, null);
                clEnqueueReadBuffer(ctx.queue(), outBuf, CL_TRUE, 0,
                    (long)Sizeof.cl_double * count, Pointer.to(results), 0, null, null);
            }

            clReleaseMemObject(posBuf);
            clReleaseMemObject(nParamBuf);
            clReleaseMemObject(octBuf);
            clReleaseMemObject(permBuf);
            clReleaseMemObject(outBuf);
            return results;
        } finally {
            ret(normalNoisePool, k);
        }
    }

    // ── evalDensityTreeFast (main chunk gen path) ─────────────────────────────

    /**
     * Dispatch density-tree evaluation. Writes the first {@code count}
     * doubles into {@code outResults} (must be at least {@code count} long).
     *
     * The caller-supplied output buffer is the perf-relevant change in
     * Phase 9.6.C: GpuDispatchQueue.dispatchGroup runs on a fixed-size pool
     * of dispatcher threads, so passing in a thread-local buffer eliminates
     * the per-call `new double[count]` that previously fronted every
     * dispatch. With ~157 ch/s × ~5× overlap × ~825 points × 8 B that was
     * ~5 MB/s of allocator churn on dispatch threads.
     */
    public void evalDensityTreeFast(double[] positions,
                                    GpuCompiledKernel gpuKernel,
                                    int count,
                                    double[] outResults) {
        if (outResults.length < count) {
            throw new IllegalArgumentException(
                "outResults too small: " + outResults.length + " < " + count);
        }
        if (count > MAX_POOLED_POINTS) {
            evalDensityTreeUnpooled(positions, gpuKernel, count, outResults);
        } else {
            evalDensityTreePooled(positions, gpuKernel, count, outResults);
        }
    }

    /**
     * Phase 9.13 — async dispatch.
     *
     * Enqueues a non-blocking write/kernel/read on a per-slot ring entry
     * and returns IMMEDIATELY. The completion thread, when the device
     * finishes the DMA read, invokes the callback with the ring entry's
     * host-side result buffer.
     *
     * Concurrency:
     *   - The dispatch thread borrows a slot, picks the next ring entry
     *     (round-robin within the slot), waits if that entry is still
     *     in-flight from a previous use (via entry.processed future),
     *     enqueues the new work non-blocking, pushes to the completion
     *     queue, and returns the slot.
     *   - The completion thread takes from the completion queue, blocks
     *     on the read event via clWaitForEvents, invokes the callback,
     *     releases the event, and completes the entry's processed future.
     *
     * Memory safety:
     *   - The Java `positions` array is held alive in entry.positionsRef
     *     until the read completes (GC root). The kernel reads from
     *     entry.posBuf, not directly from `positions` — the write DMA
     *     copies from `positions` to entry.posBuf at submission time.
     *     We use CL_FALSE (non-blocking) for the write, so the host could
     *     in theory move on before the DMA finishes; the read event
     *     transitively waits on the write through in-order queue
     *     semantics, so positions stays valid for the necessary window.
     *   - entry.hostResults is per-ring-entry, so two in-flight dispatches
     *     on the same slot don't share output buffers.
     *
     * Backpressure: when all RING_DEPTH entries of a slot are in-flight,
     * the dispatch thread waits on the next entry's processed future
     * before reusing it. With 8 slots × 4 entries = 32 in-flight max
     * before a single dispatch thread blocks; in practice the flusher
     * cycles much faster than the device, so the host catches up quickly.
     *
     * Callers must not read or write {@code positions} until the callback
     * fires — the dispatch path treats it as read-only and the DMA writes
     * its contents to GPU memory.
     */
    public void evalDensityTreeAsync(double[] positions,
                                     GpuCompiledKernel gpuKernel,
                                     int count,
                                     AsyncCallback callback) {
        if (count > MAX_POOLED_POINTS) {
            // Oversize fallback: do it synchronously, then invoke callback
            // inline. We don't expect this path on the hot dispatch flow;
            // it's preserved so an unusually large coalesced batch doesn't
            // crash. The callback runs on the calling thread.
            double[] results = new double[count];
            try {
                evalDensityTreeUnpooled(positions, gpuKernel, count, results);
                callback.onComplete(results, count);
            } catch (Throwable t) {
                callback.onFailure(t);
            }
            return;
        }

        DensitySlot slot = borrowSlot();
        boolean enqueued = false;
        try {
            // Round-robin ring index. Each slot's ring entries cycle through
            // 0,1,2,3,0,1,2,3,... per dispatch.
            int idx = slot.nextRingIndex;
            slot.nextRingIndex = (idx + 1) % RING_DEPTH;
            RingEntry entry = slot.ring[idx];

            // If this entry has been used before, the completion thread is
            // (or was) processing the previous use. Wait for it to finish.
            // entry.processed.complete(null) is called by the completion
            // thread after invoking the previous callback and releasing
            // the previous event.
            java.util.concurrent.CompletableFuture<Void> prevProcessed = entry.processed;
            if (prevProcessed != null && !prevProcessed.isDone()) {
                // Bounded wait so we never deadlock on a dropped completion.
                // 5s is generous — a single dispatch's wall is ~1 ms.
                try {
                    prevProcessed.get(5, TimeUnit.SECONDS);
                } catch (Exception e) {
                    throw new RuntimeException(
                        "Ring entry " + idx + " on slot stuck in-flight: " + e, e);
                }
            }

            // Phase 9.12.A — per-ring-entry static-arg caching. Arg 0
            // (posBuf) and arg 13 (outBuf) were bound once at slot init.
            // Args 1-12 only rebound when the GpuCompiledKernel changes
            // for this ring entry.
            cl_kernel k = entry.kernel;
            if (entry.lastGpuKernel != gpuKernel) {
                clSetKernelArg(k,  1, Sizeof.cl_mem, Pointer.to(gpuKernel.iOpsBuf));
                clSetKernelArg(k,  2, Sizeof.cl_mem, Pointer.to(gpuKernel.dArgsBuf));
                clSetKernelArg(k,  3, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseParamsBuf));
                clSetKernelArg(k,  4, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseInfoBuf));
                clSetKernelArg(k,  5, Sizeof.cl_mem, Pointer.to(gpuKernel.octaveParamsBuf));
                clSetKernelArg(k,  6, Sizeof.cl_mem, Pointer.to(gpuKernel.permTablesBuf));
                clSetKernelArg(k,  7, Sizeof.cl_mem, Pointer.to(gpuKernel.splineHeadersBuf));
                clSetKernelArg(k,  8, Sizeof.cl_mem, Pointer.to(gpuKernel.splineFloatPoolBuf));
                clSetKernelArg(k,  9, Sizeof.cl_mem, Pointer.to(gpuKernel.splineChildrenBuf));
                clSetKernelArg(k, 10, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedScalarsBuf));
                clSetKernelArg(k, 11, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinFactorsBuf));
                clSetKernelArg(k, 12, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinInfoBuf));
                entry.lastGpuKernel = gpuKernel;
                GlassPaperBenchmark.recordArgCacheMiss();
            } else {
                GlassPaperBenchmark.recordArgCacheHit();
            }
            clSetKernelArg(k, 14, Sizeof.cl_int, Pointer.to(new int[]{count}));

            // Stage in-flight state BEFORE enqueueing the read event:
            // the completion thread should see consistent fields when it
            // dequeues the PendingCompletion.
            entry.pointCount   = count;
            entry.callback     = callback;
            entry.positionsRef = positions;
            entry.processed    = new java.util.concurrent.CompletableFuture<>();

            cl_event readEv = new cl_event();
            // Non-blocking write — kernel will see the data via in-order
            // queue semantics. positions stays GC-rooted via positionsRef.
            clEnqueueWriteBuffer(slot.queue, entry.posBuf, CL_FALSE, 0,
                (long) Sizeof.cl_double * count * 3,
                Pointer.to(positions), 0, null, null);
            clEnqueueNDRangeKernel(slot.queue, k, 1, null,
                new long[]{roundUp(count, densityLocalWorkSize)},
                new long[]{densityLocalWorkSize}, 0, null, null);
            // Non-blocking read into entry.hostResults (per-ring-entry,
            // safe to write concurrent with other rings/slots). The event
            // is what the completion thread waits on.
            clEnqueueReadBuffer(slot.queue, entry.outBuf, CL_FALSE, 0,
                (long) Sizeof.cl_double * count,
                Pointer.to(entry.hostResults), 0, null, readEv);

            entry.readEvent = readEv;
            // Hand off to completion thread. After this push, the entry
            // belongs to the completion thread until it completes the
            // processed future.
            pendingCompletions.add(new PendingCompletion(entry));
            enqueued = true;
        } finally {
            returnSlot(slot);
            if (!enqueued) {
                // We borrowed the slot but failed before enqueueing.
                // Signal CPU fallback.
                try { callback.onFailure(new RuntimeException("dispatch aborted")); }
                catch (Throwable ignored) {}
            }
        }
    }

    /**
     * Allocating overload — retained so the dispatchQueue==null fallback
     * in NoiseChunk.fillSliceGpu (and any other synchronous caller) stays
     * unchanged. The hot path uses the output-array form above.
     */
    public double[] evalDensityTreeFast(double[] positions,
                                        GpuCompiledKernel gpuKernel,
                                        int count) {
        double[] results = new double[count];
        evalDensityTreeFast(positions, gpuKernel, count, results);
        return results;
    }

    /**
     * Hot path. Borrows a pre-allocated (kernel, posBuf, outBuf) slot, uploads
     * the position array via a blocking write, dispatches the kernel, then
     * blocking-reads the results. Slot buffers are sized for MAX_POOLED_POINTS
     * and reused indefinitely — no cl_mem alloc/release in this path.
     */
    private void evalDensityTreePooled(double[] positions,
                                       GpuCompiledKernel gpuKernel,
                                       int count,
                                       double[] outResults) {
        DensitySlot slot = borrowSlot();
        try {
            cl_kernel        k      = slot.kernel;
            cl_mem           posBuf = slot.posBuf;
            cl_mem           outBuf = slot.outBuf;
            cl_command_queue q      = slot.queue;

            // Phase 9.12.A — static-arg caching.
            //
            // Args 0 (posBuf) and 13 (outBuf) are bound once at slot init in
            // build() and never change for the pooled path — skip them here.
            //
            // Args 1-12 (the 12 buffer pointers from gpuKernel) only need
            // re-binding when the GpuCompiledKernel identity changes between
            // consecutive dispatches on this slot. dispatchGroup batches
            // same-kernel work items together, so within a flush wave a slot
            // may execute the same gpuKernel multiple times before being
            // reused for a different one — those repeats are pure hits.
            //
            // Across flush waves the slot pool round-robins, so hit rate
            // depends on kernel diversity vs slot count. With ~5 distinct
            // kernels per chunk slice and 8 slots, hit rate trends ~20-30%
            // in steady state per worker-thread arrival; coalescing within
            // a single dispatchGroup multiplies that.
            //
            // Each clSetKernelArg is a driver round-trip — conservative
            // estimate 1-3 µs per call on NVIDIA OpenCL. Skipping 12 calls
            // on a hit saves ~12-36 µs of host overhead per dispatch.
            if (slot.lastGpuKernel != gpuKernel) {
                clSetKernelArg(k,  1, Sizeof.cl_mem, Pointer.to(gpuKernel.iOpsBuf));
                clSetKernelArg(k,  2, Sizeof.cl_mem, Pointer.to(gpuKernel.dArgsBuf));
                clSetKernelArg(k,  3, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseParamsBuf));
                clSetKernelArg(k,  4, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseInfoBuf));
                clSetKernelArg(k,  5, Sizeof.cl_mem, Pointer.to(gpuKernel.octaveParamsBuf));
                clSetKernelArg(k,  6, Sizeof.cl_mem, Pointer.to(gpuKernel.permTablesBuf));
                clSetKernelArg(k,  7, Sizeof.cl_mem, Pointer.to(gpuKernel.splineHeadersBuf));
                clSetKernelArg(k,  8, Sizeof.cl_mem, Pointer.to(gpuKernel.splineFloatPoolBuf));
                clSetKernelArg(k,  9, Sizeof.cl_mem, Pointer.to(gpuKernel.splineChildrenBuf));
                clSetKernelArg(k, 10, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedScalarsBuf));
                clSetKernelArg(k, 11, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinFactorsBuf));
                clSetKernelArg(k, 12, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinInfoBuf));
                slot.lastGpuKernel = gpuKernel;
                GlassPaperBenchmark.recordArgCacheMiss();
            } else {
                GlassPaperBenchmark.recordArgCacheHit();
            }
            // count varies per dispatch — always set.
            clSetKernelArg(k, 14, Sizeof.cl_int, Pointer.to(new int[]{count}));

            // No mutex — the slot's queue is single-writer (this caller).
            // In-order queue semantics ensure write → kernel → read complete
            // in submission order. Different slots' queues run concurrently
            // on the device when multi-flusher submits to them in parallel.
            //
            // NOTE on validation: when many dispatchers are active, more
            // worker threads run NoiseChunk's runtime CPU/GPU validation
            // block concurrently. The CPU baseline (validateFn.compute on
            // SinglePointContext) is NOT thread-safe through MC's
            // BlendedNoise path, so `gpuvalidate` may produce false-positive
            // mismatch reports under load. Terrain itself is correct —
            // verified visually.
            //
            // Blocking write — Java GC could otherwise relocate `positions`
            // between this call and the kernel reading from it.
            //
            // Phase 9.7: when `gpuprofile` is on, allocate cl_event handles
            // for 1-in-N sampled dispatches so we can query
            // CL_PROFILING_COMMAND_START/END after the read completes and
            // accumulate per-phase nanos. Sampling avoids hammering NVIDIA's
            // per-event-tracking limit under concurrent dispatches.
            //
            // shouldProfileThisDispatch() atomically increments the counter
            // and returns true 1 in profileSampleEvery times. Concurrent
            // dispatchers never both sample the same index.
            boolean profile = GlassPaperBenchmark.shouldProfileThisDispatch();
            cl_event eWrite  = profile ? new cl_event() : null;
            cl_event eKernel = profile ? new cl_event() : null;
            cl_event eRead   = profile ? new cl_event() : null;
            long wallStart = profile ? System.nanoTime() : 0;

            clEnqueueWriteBuffer(q, posBuf, CL_TRUE, 0,
                (long) Sizeof.cl_double * count * 3,
                Pointer.to(positions), 0, null, eWrite);
            clEnqueueNDRangeKernel(q, k, 1, null,
                new long[]{roundUp(count, densityLocalWorkSize)},
                new long[]{densityLocalWorkSize}, 0, null, eKernel);
            clEnqueueReadBuffer(q, outBuf, CL_TRUE, 0,
                (long) Sizeof.cl_double * count,
                Pointer.to(outResults), 0, null, eRead);

            if (profile) {
                long wallEnd = System.nanoTime();
                // Defensive: profiling is a measurement feature. If the
                // device/driver/JOCL combo doesn't actually expose profile
                // info on our queues (CL_PROFILING_INFO_NOT_AVAILABLE), we
                // log once and disable profile mode globally — never let
                // the measurement path break the hot dispatch path.
                try {
                    long writeNs  = eventDurationNs(eWrite);
                    long kernelNs = eventDurationNs(eKernel);
                    long readNs   = eventDurationNs(eRead);
                    long queueLatencyNs = eventQueueLatencyNs(eRead);
                    GlassPaperBenchmark.recordProfile(
                        writeNs, kernelNs, readNs, queueLatencyNs, wallEnd - wallStart);
                } catch (CLException profileEx) {
                    // Don't auto-disable on individual sample failures. NVIDIA
                    // appears to lose profile-info on some events under heavy
                    // concurrent dispatch; sampled captures will still
                    // accumulate over the run. Just count the failure.
                    GlassPaperBenchmark.recordProfileFailure();
                } finally {
                    clReleaseEvent(eWrite);
                    clReleaseEvent(eKernel);
                    clReleaseEvent(eRead);
                }
            }
        } finally {
            returnSlot(slot);
        }
    }

    /**
     * CL_PROFILING_COMMAND_END - CL_PROFILING_COMMAND_START in nanoseconds.
     * Both timestamps are device-tick-converted to ns by the runtime.
     */
    private static long eventDurationNs(cl_event ev) {
        long[] start = new long[1], end = new long[1];
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
            Sizeof.cl_ulong, Pointer.to(start), null);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
            Sizeof.cl_ulong, Pointer.to(end),   null);
        return end[0] - start[0];
    }

    /** CL_PROFILING_COMMAND_SUBMIT - CL_PROFILING_COMMAND_QUEUED. */
    private static long eventQueueLatencyNs(cl_event ev) {
        long[] queued = new long[1], submit = new long[1];
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED,
            Sizeof.cl_ulong, Pointer.to(queued), null);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT,
            Sizeof.cl_ulong, Pointer.to(submit), null);
        return submit[0] - queued[0];
    }

    /**
     * Cold path. Allocates fresh cl_mem buffers per call — used only when the
     * request exceeds MAX_POOLED_POINTS. This is the pre-Phase-9.4 behavior.
     */
    private void evalDensityTreeUnpooled(double[] positions,
                                         GpuCompiledKernel gpuKernel,
                                         int count,
                                         double[] outResults) {
        DensitySlot slot = borrowSlot();
        try {
            cl_kernel        k = slot.kernel;
            cl_command_queue q = slot.queue;
            // Size by count*3, not positions.length: thread-local position
            // buffers in GpuDispatchQueue are grown with headroom and may
            // be larger than the actual point count for this dispatch.
            cl_mem posBuf = clCreateBuffer(ctx.context(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_double * count * 3,
                Pointer.to(positions), null);
            cl_mem outBuf = clCreateBuffer(ctx.context(),
                CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_double * count,
                null, null);

            clSetKernelArg(k,  0, Sizeof.cl_mem, Pointer.to(posBuf));
            clSetKernelArg(k,  1, Sizeof.cl_mem, Pointer.to(gpuKernel.iOpsBuf));
            clSetKernelArg(k,  2, Sizeof.cl_mem, Pointer.to(gpuKernel.dArgsBuf));
            clSetKernelArg(k,  3, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseParamsBuf));
            clSetKernelArg(k,  4, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseInfoBuf));
            clSetKernelArg(k,  5, Sizeof.cl_mem, Pointer.to(gpuKernel.octaveParamsBuf));
            clSetKernelArg(k,  6, Sizeof.cl_mem, Pointer.to(gpuKernel.permTablesBuf));
            clSetKernelArg(k,  7, Sizeof.cl_mem, Pointer.to(gpuKernel.splineHeadersBuf));
            clSetKernelArg(k,  8, Sizeof.cl_mem, Pointer.to(gpuKernel.splineFloatPoolBuf));
            clSetKernelArg(k,  9, Sizeof.cl_mem, Pointer.to(gpuKernel.splineChildrenBuf));
            clSetKernelArg(k, 10, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedScalarsBuf));
            clSetKernelArg(k, 11, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinFactorsBuf));
            clSetKernelArg(k, 12, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinInfoBuf));
            clSetKernelArg(k, 13, Sizeof.cl_mem, Pointer.to(outBuf));
            clSetKernelArg(k, 14, Sizeof.cl_int, Pointer.to(new int[]{count}));

            clEnqueueNDRangeKernel(q, k, 1, null,
                new long[]{roundUp(count, densityLocalWorkSize)},
                new long[]{densityLocalWorkSize}, 0, null, null);
            clEnqueueReadBuffer(q, outBuf, CL_TRUE, 0,
                (long) Sizeof.cl_double * count, Pointer.to(outResults), 0, null, null);

            clReleaseMemObject(posBuf);
            clReleaseMemObject(outBuf);

            // Phase 9.12.A — this path just clobbered arg 0 and arg 13 with
            // local temp buffers that we're about to release. Restore the
            // slot's persistent buffer bindings, and invalidate the per-slot
            // gpuKernel cache so the next pooled dispatch re-binds args 1-12
            // (this path also rebound those args, but lastGpuKernel wasn't
            // updated, so without this it'd skip the rebind and use stale
            // pointers for one dispatch).
            clSetKernelArg(slot.kernel,  0, Sizeof.cl_mem, Pointer.to(slot.posBuf));
            clSetKernelArg(slot.kernel, 13, Sizeof.cl_mem, Pointer.to(slot.outBuf));
            slot.lastGpuKernel = null;
        } finally {
            returnSlot(slot);
        }
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    private static long roundUp(int n, int multiple) {
        return (long)(((n + multiple - 1) / multiple) * multiple);
    }

    public void release() {
        // Phase 9.13 — drain in-flight async dispatches before releasing
        // GPU resources. Stop the completion thread, then process any
        // remaining pending completions inline so worker futures aren't
        // orphaned.
        completionRunning = false;
        if (completionThread != null) {
            try {
                completionThread.join(2000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
        // Drain anything left in the queue. The thread may have exited
        // mid-loop with items still queued.
        PendingCompletion leftover;
        while ((leftover = pendingCompletions.poll()) != null) {
            RingEntry entry = leftover.entry;
            cl_event ev = entry.readEvent;
            try {
                if (ev != null) clWaitForEvents(1, new cl_event[]{ev});
                if (entry.callback != null) {
                    try { entry.callback.onComplete(entry.hostResults, entry.pointCount); }
                    catch (Throwable ignored) {}
                }
            } catch (Throwable ignored) {
            } finally {
                if (ev != null) {
                    try { clReleaseEvent(ev); } catch (Throwable ignored) {}
                }
                java.util.concurrent.CompletableFuture<Void> f = entry.processed;
                if (f != null) f.complete(null);
            }
        }

        for (cl_kernel k : noisePool)       clReleaseKernel(k);
        for (cl_kernel k : normalNoisePool) clReleaseKernel(k);
        for (DensitySlot s : densityTreeSlotPool) {
            // Sync path resources
            clReleaseKernel(s.kernel);
            clReleaseMemObject(s.posBuf);
            clReleaseMemObject(s.outBuf);
            // Phase 9.13 — ring entry resources
            for (RingEntry e : s.ring) {
                clReleaseKernel(e.kernel);
                clReleaseMemObject(e.posBuf);
                clReleaseMemObject(e.outBuf);
            }
        }
        clReleaseProgram(program);
    }
}
