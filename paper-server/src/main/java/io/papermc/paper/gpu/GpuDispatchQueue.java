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
        int total = pendingPoints.addAndGet(count);

        // Wake flusher immediately if threshold reached
        if (total >= flushThreshold) {
            synchronized (flusherThread) {
                flusherThread.notifyAll();
            }
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
            // Wait until threshold reached or interval elapsed
            synchronized (flusherThread) {
                if (pendingPoints.get() < flushThreshold) {
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
     * array for one kernel group, performs the GPU dispatch (which borrows
     * a DensitySlot internally), and routes results back to each work item's
     * future. On failure, completes each future with null so the calling
     * worker can fall back to CPU evaluation.
     */
    private void dispatchGroup(GpuCompiledKernel ck, List<GpuWorkItem> group) {
        // Build merged position array
        int totalPoints = 0;
        for (GpuWorkItem w : group) {
            w.batchOffset = totalPoints;
            totalPoints  += w.count;
        }

        double[] mergedPositions = new double[totalPoints * 3];
        for (GpuWorkItem w : group) {
            System.arraycopy(w.positions, 0,
                mergedPositions, w.batchOffset * 3, w.count * 3);
        }

        // Single GPU dispatch for the whole group
        double[] mergedResults;
        try {
            long start = System.nanoTime();
            mergedResults = kernel.evalDensityTreeFast(mergedPositions, ck, totalPoints);
            GlassPaperBenchmark.recordGpu(System.nanoTime() - start, totalPoints);
            GlassPaperBenchmark.recordBatchFlush(totalPoints);
        } catch (Exception e) {
            LOGGER.warning("GPU batch dispatch failed: " + e.getMessage());
            // Signal CPU fallback to each waiting worker. Zero-filling the
            // result here would silently corrupt terrain — the caller
            // interprets a non-null array as a successful GPU dispatch and
            // copies it straight into the interpolator slice.
            for (GpuWorkItem w : group) {
                w.future.complete(null);
            }
            return;
        }

        // Split results back to individual futures
        for (GpuWorkItem w : group) {
            double[] slice = new double[w.count];
            System.arraycopy(mergedResults, w.batchOffset, slice, 0, w.count);
            w.future.complete(slice);
        }
    }
}
