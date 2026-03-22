package io.papermc.paper.gpu;

import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Lightweight lock-free benchmark collector for GlassPaper GPU timing.
 * Call report() to print a summary to the server log.
 */
public final class GlassPaperBenchmark {

    private static final Logger LOGGER = Logger.getLogger("GlassPaper");

    // GPU stats
    private static final AtomicLong gpuCalls       = new AtomicLong();
    private static final AtomicLong gpuTotalNanos   = new AtomicLong();
    private static final AtomicLong gpuTotalPoints  = new AtomicLong();

    // CPU fallback stats (for non-Noise density functions)
    private static final AtomicLong cpuCalls        = new AtomicLong();
    private static final AtomicLong cpuTotalNanos   = new AtomicLong();

    private GlassPaperBenchmark() {}

    public static void recordGpu(long nanos, int points) {
        gpuCalls.incrementAndGet();
        gpuTotalNanos.addAndGet(nanos);
        gpuTotalPoints.addAndGet(points);
    }

    public static void recordCpu(long nanos) {
        cpuCalls.incrementAndGet();
        cpuTotalNanos.addAndGet(nanos);
    }

    public static void report() {
        long gCalls  = gpuCalls.get();
        long gNanos  = gpuTotalNanos.get();
        long gPoints = gpuTotalPoints.get();
        long cCalls  = cpuCalls.get();
        long cNanos  = cpuTotalNanos.get();

        if (gCalls == 0 && cCalls == 0) {
            LOGGER.info("[GlassPaper] No benchmark data collected yet. Generate some chunks first.");
            return;
        }

        LOGGER.info("[GlassPaper] ════════════════════════════════════════");
        LOGGER.info("[GlassPaper]        GlassPaper Benchmark Report      ");
        LOGGER.info("[GlassPaper] ════════════════════════════════════════");

        if (gCalls > 0) {
            double avgGpuMs   = (gNanos / (double) gCalls) / 1_000_000.0;
            double totalGpuMs = gNanos / 1_000_000.0;
            double pointsPerMs = gPoints / (gNanos / 1_000_000.0);
            LOGGER.info(String.format("[GlassPaper] GPU dispatches  : %,d", gCalls));
            LOGGER.info(String.format("[GlassPaper] GPU total time  : %.2f ms", totalGpuMs));
            LOGGER.info(String.format("[GlassPaper] GPU avg/dispatch: %.4f ms", avgGpuMs));
            LOGGER.info(String.format("[GlassPaper] GPU throughput  : %.0f samples/ms", pointsPerMs));
            LOGGER.info(String.format("[GlassPaper] GPU total points: %,d", gPoints));
        } else {
            LOGGER.info("[GlassPaper] GPU dispatches  : 0 (no Noise functions hit GPU path)");
        }

        if (cCalls > 0) {
            double avgCpuMs   = (cNanos / (double) cCalls) / 1_000_000.0;
            double totalCpuMs = cNanos / 1_000_000.0;
            LOGGER.info(String.format("[GlassPaper] CPU fallbacks   : %,d", cCalls));
            LOGGER.info(String.format("[GlassPaper] CPU total time  : %.2f ms", totalCpuMs));
            LOGGER.info(String.format("[GlassPaper] CPU avg/call    : %.4f ms", avgCpuMs));
        } else {
            LOGGER.info("[GlassPaper] CPU fallbacks   : 0");
        }

        LOGGER.info("[GlassPaper] ════════════════════════════════════════");

        if (!functionTypes.isEmpty()) {
            LOGGER.info("[GlassPaper] Density function types seen by interpolators:");
            functionTypes.forEach((type, count) ->
                LOGGER.info(String.format("[GlassPaper]   %s : %,d", type, count.get())));
        }
    }

    public static void reset() {
        gpuCalls.set(0);
        gpuTotalNanos.set(0);
        gpuTotalPoints.set(0);
        cpuCalls.set(0);
        cpuTotalNanos.set(0);
        LOGGER.info("[GlassPaper] Benchmark counters reset.");
    }

    private static final java.util.concurrent.ConcurrentHashMap<String, AtomicLong> functionTypes
        = new java.util.concurrent.ConcurrentHashMap<>();

    public static void recordFunctionType(String type) {
        functionTypes.computeIfAbsent(type, k -> new AtomicLong()).incrementAndGet();
    }

}
