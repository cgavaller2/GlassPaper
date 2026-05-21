package io.papermc.paper.gpu;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

/**
 * Accumulates GPU work items from all worker threads and flushes them
 * in large batches to amortize per-dispatch overhead.
 *
 * Flush triggers (configurable):
 *   - pendingPoints >= flushThreshold
 *   - flushIntervalMs elapsed since last flush
 *
 * Worker threads block on GpuWorkItem.future.get() until the flusher
 * completes their item.
 */
public final class GpuDispatchQueue {

    private static final Logger LOGGER = Logger.getLogger("GlassPaper");

    // ── Singleton ─────────────────────────────────────────────────────────────
    private static volatile GpuDispatchQueue INSTANCE;

    public static GpuDispatchQueue get()           { return INSTANCE; }
    public static void set(GpuDispatchQueue queue) { INSTANCE = queue; }

    // ── Config (tunable after benchmarks) ────────────────────────────────────
    private volatile int  flushThreshold;   // flush when pendingPoints >= this
    private volatile long flushIntervalMs;  // flush at least every N ms

    // ── State ─────────────────────────────────────────────────────────────────
    private final ConcurrentLinkedQueue<GpuWorkItem> pendingQueue =
        new ConcurrentLinkedQueue<>();
    private final AtomicInteger pendingPoints = new AtomicInteger(0);

    private final GpuNoiseKernel kernel;
    private final Thread flusherThread;
    private volatile boolean running = true;

    // Phase 9.5 — multi-flusher dispatch pool. The coordinator thread
    // (flusherThread) drains the pending queue and groups work by kernel,
    // then submits each group as a parallel task. Up to DISPATCH_POOL_SIZE
    // groups dispatch concurrently, each using one of the per-slot OpenCL
    // queues from GpuNoiseKernel. Matches the slot/queue count so we never
    // contend on the slot pool.
    private static final int DISPATCH_POOL_SIZE = GpuContext.DENSITY_QUEUE_COUNT;
    private final ExecutorService dispatchPool;

    // ── Constructor ───────────────────────────────────────────────────────────

    public GpuDispatchQueue(GpuNoiseKernel kernel, int flushThreshold, long flushIntervalMs) {
        this.kernel           = kernel;
        this.flushThreshold   = flushThreshold;
        this.flushIntervalMs  = flushIntervalMs;

        // Daemon-thread executor so JVM shutdown isn't blocked.
        final AtomicInteger threadId = new AtomicInteger();
        this.dispatchPool = Executors.newFixedThreadPool(DISPATCH_POOL_SIZE, r -> {
            Thread t = new Thread(r, "GlassPaper-GPU-Dispatch-" + threadId.getAndIncrement());
            t.setDaemon(true);
            t.setPriority(Thread.NORM_PRIORITY + 1);
            return t;
        });

        this.flusherThread = new Thread(this::flusherLoop, "GlassPaper-GPU-Flusher");
        this.flusherThread.setDaemon(true);
        this.flusherThread.setPriority(Thread.NORM_PRIORITY + 1);
        this.flusherThread.start();

        LOGGER.info(String.format(
            "GPU dispatch queue started (threshold=%d points, interval=%dms, "
          + "dispatch pool=%d)",
            flushThreshold, flushIntervalMs, DISPATCH_POOL_SIZE));
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Submit work and block until the GPU result is ready.
     * Convenience wrapper around {@link #submitAsync} + {@link #awaitResult}.
     */
    public double[] submit(double[] positions, GpuCompiledKernel gpuKernel, int count) {
        return awaitResult(submitAsync(positions, gpuKernel, count));
    }

    /**
     * Submit work without blocking. Returns the work item; the caller is
     * responsible for awaiting its future via {@link #awaitResult}.
     *
     * Use this to enqueue multiple submissions (e.g. all interpolators of a
     * slice fill) before any blocking, so the flusher sees more points per
     * batch and can coalesce them into a single GPU dispatch.
     */
    public GpuWorkItem submitAsync(double[] positions, GpuCompiledKernel gpuKernel, int count) {
        GpuWorkItem item = new GpuWorkItem(positions, gpuKernel, count);
        pendingQueue.add(item);
        pendingPoints.addAndGet(count);

        // Phase 9.13.B — always notify the flusher on submit. Previously
        // we only notified when pendingPoints crossed the threshold, on
        // the theory that the threshold-or-interval wait was the cycle
        // pacer. With async dispatch (Phase 9.13) the dispatch path no
        // longer blocks, so the flusher can cycle much faster than 1/ms.
        // Notifying on every submit lets it pick up work the moment it
        // arrives. The flusherLoop also now skips its 1ms interval wait
        // entirely when pendingPoints > 0, so the actual flush cadence
        // becomes bounded by enqueue time (~100 µs) rather than the
        // interval. This drives ring depth utilization up from ~12% (one
        // wave per ms × 5 dispatches) to whatever the workers can sustain.
        synchronized (flusherThread) {
            flusherThread.notifyAll();
        }
        return item;
    }

    /**
     * Block until the work item completes. Returns null on timeout, on
     * dispatch exception (flusher completes with null), or on shutdown.
     * In every null case the caller must fall back to CPU.
     */
    public double[] awaitResult(GpuWorkItem item) {
        try {
            return item.future.get(50, java.util.concurrent.TimeUnit.MILLISECONDS);
        } catch (java.util.concurrent.TimeoutException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Update flush parameters at runtime without restart.
     */
    public void configure(int newThreshold, long newIntervalMs) {
        this.flushThreshold  = newThreshold;
        this.flushIntervalMs = newIntervalMs;
        LOGGER.info(String.format(
            "GPU dispatch queue reconfigured (threshold=%d, interval=%dms)",
            newThreshold, newIntervalMs));
    }

    public int getFlushThreshold()  { return flushThreshold;  }
    public long getFlushIntervalMs(){ return flushIntervalMs; }

    /** Flush remaining work and stop the flusher thread + dispatch pool. */
    public void shutdown() {
        running = false;
        synchronized (flusherThread) {
            flusherThread.notifyAll();
        }
        try {
            flusherThread.join(2000);
        } catch (InterruptedException ignored) {}

        dispatchPool.shutdown();
        try {
            if (!dispatchPool.awaitTermination(2, TimeUnit.SECONDS)) {
                dispatchPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            dispatchPool.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // Unblock waiting workers with a CPU-fallback signal — null tells
        // submit() to return null so the caller runs the work on CPU.
        GpuWorkItem item;
        while ((item = pendingQueue.poll()) != null) {
            item.future.complete(null);
        }
    }

    // ── Flusher loop ──────────────────────────────────────────────────────────

    private void flusherLoop() {
        while (running) {
            // Phase 9.13.B — only wait when there is genuinely no work.
            // Previously we waited up to flushIntervalMs (1ms) whenever
            // pendingPoints was below threshold, which pinned flush rate
            // to ~1/ms even when work was constantly available. With
            // async dispatch the flush itself is cheap (~100 µs to enqueue
            // 5 dispatches into the rings), so we should flush whenever
            // there's anything to flush.
            //
            // The wait condition is now "queue empty" — pendingPoints == 0
            // is the only state where waiting helps (no work to flush).
            // If pendingPoints > 0, skip wait, go flush. After flush(),
            // loop back; if more work has arrived (or completion thread
            // unblocked workers who submitted more), we flush immediately
            // without waiting.
            //
            // The flushIntervalMs is kept as the wait timeout so we still
            // wake periodically to check `running` and handle shutdown
            // cleanly even with no submit-side notify firing.
            synchronized (flusherThread) {
                if (pendingPoints.get() == 0) {
                    try {
                        flusherThread.wait(flushIntervalMs);
                    } catch (InterruptedException ignored) {}
                }
            }

            if (pendingPoints.get() > 0) {
                flush();
            }
        }
        // Final flush on shutdown
        if (pendingPoints.get() > 0) flush();
    }

    private void flush() {
        // ── 1. Drain the queue ───────────────────────────────────────────────
        List<GpuWorkItem> batch = new ArrayList<>();
        GpuWorkItem item;
        while ((item = pendingQueue.poll()) != null) {
            batch.add(item);
        }
        if (batch.isEmpty()) return;

        int totalDrained = 0;
        for (GpuWorkItem w : batch) totalDrained += w.count;
        pendingPoints.addAndGet(-totalDrained);

        // ── 2. Group by kernel identity (usually 2-4 unique kernels) ─────────
        Map<GpuCompiledKernel, List<GpuWorkItem>> byKernel = new IdentityHashMap<>();
        for (GpuWorkItem w : batch) {
            byKernel.computeIfAbsent(w.gpuKernel, k -> new ArrayList<>()).add(w);
        }

        // ── 3. Parallel GPU dispatch per kernel group ───────────────────────
        // Phase 9.5 — submit each kernel group to the dispatch pool. Each
        // worker borrows its own DensitySlot (with a dedicated cl_command_queue
        // and pre-allocated posBuf/outBuf), so up to DISPATCH_POOL_SIZE groups
        // run concurrently on the device. We wait for all groups in this
        // batch to complete before returning, which keeps flush boundaries
        // clean and bounds the in-flight backlog.
        //
        // Earlier pipelining attempts on a SINGLE queue (OOO + cl_event
        // chains, non-blocking reads + clFinish) hung or timed out. This
        // design avoids both: per-queue ops are in-order and self-blocking,
        // and parallelism comes from the thread pool, not from event chains.
        List<Future<?>> dispatchTasks = new ArrayList<>(byKernel.size());
        for (Map.Entry<GpuCompiledKernel, List<GpuWorkItem>> entry : byKernel.entrySet()) {
            final GpuCompiledKernel ck    = entry.getKey();
            final List<GpuWorkItem> group = entry.getValue();
            dispatchTasks.add(dispatchPool.submit(() -> dispatchGroup(ck, group)));
        }

        // Wait for all parallel dispatches in this flush to complete.
        // Bounded timeout in case any dispatch hangs (driver bug, OOM) so
        // we don't deadlock the flusher thread.
        for (Future<?> task : dispatchTasks) {
            try {
                task.get(5000, TimeUnit.MILLISECONDS);
            } catch (Exception e) {
                LOGGER.warning("GPU dispatch task failed/timed out: " + e.getMessage());
                // Task ran on the dispatch pool — its group's futures were
                // either completed normally or with null inside dispatchGroup,
                // so workers are not stuck. We just log and move on.
            }
        }
    }

    /**
     * Runs on a dispatch-pool worker thread. Builds the merged position
     * array for one kernel group, ENQUEUES the GPU dispatch via the async
     * API (Phase 9.13), and returns immediately. The completion thread
     * inside {@link GpuNoiseKernel} will invoke the callback to split
     * merged GPU results back to per-item futures when the device finishes.
     *
     * On enqueue failure, completes each waiting worker's future with null
     * so they fall back to CPU.
     *
     * Phase 9.13 architectural change: this method no longer blocks on
     * clEnqueueReadBuffer. The dispatch-pool thread frees up almost
     * immediately (~50 µs of host-side enqueue overhead per call instead
     * of the prior ~1 ms wall). The flusher's wait-on-tasks loop now only
     * waits for the ENQUEUE to finish, not the device-side completion —
     * which means many more flushes per unit time and more device pipeline
     * depth. Worker futures are completed by the completion thread inside
     * GpuNoiseKernel rather than this method.
     *
     * Per-merged-position-buffer is allocated fresh each call now (vs the
     * thread-local reuse in Phase 9.6.D), because the buffer must remain
     * GC-rooted until the async DMA finishes — a thread-local would be
     * overwritten by the next dispatch on the same thread before the prior
     * device read completes. The async API holds positionsRef internally
     * on the RingEntry, so the GC can reclaim the merged buffer once the
     * read completes and the entry's positionsRef is cleared.
     */
    private void dispatchGroup(GpuCompiledKernel ck, List<GpuWorkItem> group) {
        // Build merged position array
        int totalPoints = 0;
        for (GpuWorkItem w : group) {
            w.batchOffset = totalPoints;
            totalPoints  += w.count;
        }
        // Stack-local captured by the callback closure; must not be reused
        // by this thread or the next dispatch until the device has read it.
        final double[] mergedPositions = new double[totalPoints * 3];
        for (GpuWorkItem w : group) {
            System.arraycopy(w.positions, 0,
                mergedPositions, w.batchOffset * 3, w.count * 3);
        }

        final int finalTotalPoints = totalPoints;
        final List<GpuWorkItem> finalGroup = group;
        final long enqueueStart = System.nanoTime();

        // Async dispatch — the callback runs on the completion thread when
        // the device's read DMA finishes. From this dispatch-pool thread's
        // perspective, evalDensityTreeAsync returns after the (non-blocking)
        // enqueue calls (~50-100 µs of host overhead) instead of waiting
        // for the kernel to actually finish (~1 ms wall).
        GpuNoiseKernel.AsyncCallback cb = new GpuNoiseKernel.AsyncCallback() {
            @Override
            public void onComplete(double[] hostResults, int pointCount) {
                // The benchmark stats: device-side time is best measured by
                // the difference between enqueue start (host) and completion
                // (also host, at this callback). Includes both host enqueue
                // overhead AND the device-side execution time — this is the
                // honest "wall time" per dispatch from the consumer's POV.
                GlassPaperBenchmark.recordGpu(
                    System.nanoTime() - enqueueStart, pointCount);
                GlassPaperBenchmark.recordBatchFlush(pointCount);
                // Split per-item slices and complete worker futures
                for (GpuWorkItem w : finalGroup) {
                    double[] slice = new double[w.count];
                    System.arraycopy(hostResults, w.batchOffset, slice, 0, w.count);
                    w.future.complete(slice);
                }
            }
            @Override
            public void onFailure(Throwable cause) {
                LOGGER.warning("GPU async dispatch failed: " + cause.getMessage());
                for (GpuWorkItem w : finalGroup) {
                    w.future.complete(null);
                }
            }
        };

        try {
            kernel.evalDensityTreeAsync(mergedPositions, ck, finalTotalPoints, cb);
        } catch (Exception e) {
            LOGGER.warning("GPU async enqueue failed: " + e.getMessage());
            for (GpuWorkItem w : finalGroup) {
                w.future.complete(null);
            }
        }
        // Returns immediately; dispatch thread is free for the next group.
    }
}
