package io.papermc.paper.gpu;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
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

    // ── Constructor ───────────────────────────────────────────────────────────

    public GpuDispatchQueue(GpuNoiseKernel kernel, int flushThreshold, long flushIntervalMs) {
        this.kernel           = kernel;
        this.flushThreshold   = flushThreshold;
        this.flushIntervalMs  = flushIntervalMs;

        this.flusherThread = new Thread(this::flusherLoop, "GlassPaper-GPU-Flusher");
        this.flusherThread.setDaemon(true);
        this.flusherThread.setPriority(Thread.NORM_PRIORITY + 1);
        this.flusherThread.start();

        LOGGER.info(String.format(
            "GPU dispatch queue started (threshold=%d points, interval=%dms)",
            flushThreshold, flushIntervalMs));
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

    /** Flush remaining work and stop the flusher thread. */
    public void shutdown() {
        running = false;
        synchronized (flusherThread) {
            flusherThread.notifyAll();
        }
        try {
            flusherThread.join(2000);
        } catch (InterruptedException ignored) {}
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

        // ── 3. One blocking GPU dispatch per kernel group ───────────────────
        // Pipelining experiments (async enqueue, OOO queue, clFinish-at-end)
        // failed in practice on NVIDIA + JOCL — either events hang or all
        // dispatches time out. Stick with the simple per-group blocking
        // dispatch; the meaningful coalescing happens at the queue layer
        // (work items for the same kernel merge into one big position array).
        for (Map.Entry<GpuCompiledKernel, List<GpuWorkItem>> entry : byKernel.entrySet()) {
            GpuCompiledKernel ck    = entry.getKey();
            List<GpuWorkItem> group = entry.getValue();

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
                continue;
            }

            // Split results back to individual futures
            for (GpuWorkItem w : group) {
                double[] slice = new double[w.count];
                System.arraycopy(mergedResults, w.batchOffset, slice, 0, w.count);
                w.future.complete(slice);
            }
        }
    }
}
