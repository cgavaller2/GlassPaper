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

    private static final AtomicLong batchFlushes     = new AtomicLong();
    private static final AtomicLong batchTotalPoints  = new AtomicLong();
    private static long             largestBatch      = 0;
    private static final Object     batchLock         = new Object();

    private static final AtomicLong blendedFallbacks = new AtomicLong();

    public static void recordBlendedFallback() { blendedFallbacks.incrementAndGet(); }

    private GlassPaperBenchmark() {}

    private static volatile boolean validating = false;
    private static final java.util.concurrent.ConcurrentHashMap<String, AtomicLong>
        mismatches = new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.concurrent.ConcurrentHashMap<String, AtomicLong>
        samplesChecked = new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.concurrent.ConcurrentHashMap<String, MismatchStat>
        perTypeStats = new java.util.concurrent.ConcurrentHashMap<>();
    private static volatile double maxMismatch = 0;

    /** Per-density-function-type mismatch tracking. Mutations guarded on the instance. */
    static final class MismatchStat {
        double maxDelta;
        int    maxX, maxY, maxZ;
        double maxCpu, maxGpu;
        double sumDelta;
        long   count;
    }

    public static boolean isValidating() { return validating; }
    public static void setValidating(boolean v) { validating = v; }

    public static void recordSampleChecked(String type) {
        samplesChecked.computeIfAbsent(type, k -> new AtomicLong()).incrementAndGet();
    }

    /** Limit on rich-diagnosed mismatches to keep logs readable. */
    private static final int  MAX_RICH_DIAGNOSES = 10;
    private static final AtomicLong richDiagnosed = new AtomicLong();

    /**
     * For RangeChoice mismatches, reflect into the density function and log
     * the input value, both branch values on CPU, and which branch the GPU
     * output most likely matches. Distinguishes:
     *   wrong-branch  : GPU's RangeChoice input differs from CPU's → upstream bug
     *   wrong-value   : GPU picked the same branch but computed it wrong → branch bug
     */
    public static void diagnoseRangeChoice(Object fn,
                                           double px, double py, double pz,
                                           double cpuVal, double gpuVal) {
        if (richDiagnosed.get() >= MAX_RICH_DIAGNOSES) return;
        try {
            Class<?> c = fn.getClass();
            if (!c.getSimpleName().equals("RangeChoice")) return;
            java.lang.reflect.Field fInput = c.getDeclaredField("input");
            java.lang.reflect.Field fMin   = c.getDeclaredField("minInclusive");
            java.lang.reflect.Field fMax   = c.getDeclaredField("maxExclusive");
            java.lang.reflect.Field fIn    = c.getDeclaredField("whenInRange");
            java.lang.reflect.Field fOut   = c.getDeclaredField("whenOutOfRange");
            fInput.setAccessible(true); fMin.setAccessible(true);
            fMax.setAccessible(true); fIn.setAccessible(true); fOut.setAccessible(true);

            Object input    = fInput.get(fn);
            double min      = (double) fMin.get(fn);
            double max      = (double) fMax.get(fn);
            Object inRange  = fIn.get(fn);
            Object outRange = fOut.get(fn);

            net.minecraft.world.level.levelgen.DensityFunction.SinglePointContext ctx =
                new net.minecraft.world.level.levelgen.DensityFunction.SinglePointContext(
                    (int) px, (int) py, (int) pz);

            double inputVal =
                ((net.minecraft.world.level.levelgen.DensityFunction) input).compute(ctx);
            double inVal =
                ((net.minecraft.world.level.levelgen.DensityFunction) inRange).compute(ctx);
            double outVal =
                ((net.minecraft.world.level.levelgen.DensityFunction) outRange).compute(ctx);

            boolean cpuPicksIn = (inputVal >= min && inputVal < max);
            String  cpuChoice  = cpuPicksIn ? "IN" : "OUT";

            // Try to infer which branch GPU took by which value it matches
            double dToIn  = Math.abs(gpuVal - inVal);
            double dToOut = Math.abs(gpuVal - outVal);
            String gpuChoice = dToIn < dToOut
                ? String.format("IN  (matches in to %.3e)", dToIn)
                : String.format("OUT (matches out to %.3e)", dToOut);

            long n = richDiagnosed.incrementAndGet();
            LOGGER.warning(String.format(
                "RangeChoice diagnosis #%d @(%.0f,%.0f,%.0f):\n" +
                "    input    = %.6f   range=[%.3f, %.3f)\n" +
                "    in-range = %.6f   out-range = %.6f\n" +
                "    cpu picks %s -> %.6f\n" +
                "    gpu output = %.6f -> %s\n" +
                "    delta = %.3e",
                n, px, py, pz,
                inputVal, min, max,
                inVal, outVal,
                cpuChoice, cpuVal,
                gpuVal, gpuChoice,
                Math.abs(cpuVal - gpuVal)));
        } catch (Throwable t) {
            // First failure only — don't spam if reflection breaks
            if (richDiagnosed.compareAndSet(0, MAX_RICH_DIAGNOSES)) {
                LOGGER.warning("RangeChoice reflection failed: " + t.getMessage());
            }
        }
    }

    public static void recordMismatch(String type, double delta, int x, int y, int z) {
        recordMismatch(type, delta, x, y, z, Double.NaN, Double.NaN);
    }

    /**
     * Record a CPU/GPU divergence at a specific position. Keeps the largest
     * delta per type (and the position that produced it) for the report.
     */
    public static void recordMismatch(String type, double delta,
                                      int x, int y, int z, double cpu, double gpu) {
        mismatches.computeIfAbsent(type, k -> new AtomicLong()).incrementAndGet();
        MismatchStat s = perTypeStats.computeIfAbsent(type, k -> new MismatchStat());
        synchronized (s) {
            s.count++;
            s.sumDelta += delta;
            if (delta > s.maxDelta) {
                s.maxDelta = delta;
                s.maxX = x; s.maxY = y; s.maxZ = z;
                s.maxCpu = cpu; s.maxGpu = gpu;
            }
        }
        synchronized (GlassPaperBenchmark.class) {
            if (delta > maxMismatch) maxMismatch = delta;
        }
    }

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
            LOGGER.info("No benchmark data collected yet. Generate some chunks first.");
            return;
        }

        LOGGER.info("════════════════════════════════════════");
        LOGGER.info("       GlassPaper Benchmark Report      ");
        LOGGER.info("════════════════════════════════════════");

        long bf = batchFlushes.get();
        if (bf > 0) {
            LOGGER.info(String.format("Batch flushes    : %,d", bf));
            LOGGER.info(String.format("Avg batch size   : %.0f points",
                batchTotalPoints.get() / (double) bf));
            synchronized (batchLock) {
                LOGGER.info(String.format("Largest batch    : %,d points", largestBatch));
            }
        }

        LOGGER.info(String.format("Blended fallbacks: %,d", blendedFallbacks.get()));

        if (!compileFailures.isEmpty()) {
            LOGGER.info("Compile failures by reason:");
            compileFailures.forEach((reason, count) ->
                LOGGER.info(String.format("  %s : %,d", reason, count.get())));
        }

        if (gCalls > 0) {
            double avgGpuMs   = (gNanos / (double) gCalls) / 1_000_000.0;
            double totalGpuMs = gNanos / 1_000_000.0;
            double pointsPerMs = gPoints / (gNanos / 1_000_000.0);
            LOGGER.info(String.format("GPU dispatches  : %,d", gCalls));
            LOGGER.info(String.format("GPU total time  : %.2f ms", totalGpuMs));
            LOGGER.info(String.format("GPU avg/dispatch: %.4f ms", avgGpuMs));
            LOGGER.info(String.format("GPU throughput  : %.0f samples/ms", pointsPerMs));
            LOGGER.info(String.format("GPU total points: %,d", gPoints));
        } else {
            LOGGER.info("GPU dispatches  : 0 (no Noise functions hit GPU path)");
        }

        if (cCalls > 0) {
            double avgCpuMs   = (cNanos / (double) cCalls) / 1_000_000.0;
            double totalCpuMs = cNanos / 1_000_000.0;
            LOGGER.info(String.format("CPU fallbacks   : %,d", cCalls));
            LOGGER.info(String.format("CPU total time  : %.2f ms", totalCpuMs));
            LOGGER.info(String.format("CPU avg/call    : %.4f ms", avgCpuMs));
        } else {
            LOGGER.info("CPU fallbacks   : 0");
        }

        LOGGER.info("════════════════════════════════════════");

        if (!functionTypes.isEmpty()) {
            LOGGER.info("Density function types seen by interpolators:");
            functionTypes.forEach((type, count) ->
                LOGGER.info(String.format("  %s : %,d", type, count.get())));
        }

        if (!samplesChecked.isEmpty() || !mismatches.isEmpty()) {
            LOGGER.info("─── Runtime CPU/GPU validation ───");
            LOGGER.info(String.format("Max delta (global) : %.6e", maxMismatch));
            // Sort by maxDelta descending — worst offenders first
            java.util.List<java.util.Map.Entry<String, MismatchStat>> sorted =
                new java.util.ArrayList<>(perTypeStats.entrySet());
            sorted.sort((a, b) -> Double.compare(b.getValue().maxDelta, a.getValue().maxDelta));
            for (java.util.Map.Entry<String, MismatchStat> e : sorted) {
                String type = e.getKey();
                MismatchStat s = e.getValue();
                long checked = samplesChecked.getOrDefault(type, new AtomicLong()).get();
                synchronized (s) {
                    double avg = s.count > 0 ? s.sumDelta / s.count : 0.0;
                    double rate = checked > 0 ? (s.count * 100.0 / checked) : 0.0;
                    LOGGER.info(String.format(
                        "  %-26s %,8d / %,8d (%.1f%%)  max=%.3e avg=%.3e  worst@(%d,%d,%d) cpu=%.6f gpu=%.6f",
                        type, s.count, checked, rate, s.maxDelta, avg,
                        s.maxX, s.maxY, s.maxZ, s.maxCpu, s.maxGpu));
                }
            }
            // Types that were checked but had zero mismatches — still worth knowing
            for (java.util.Map.Entry<String, AtomicLong> e : samplesChecked.entrySet()) {
                if (!perTypeStats.containsKey(e.getKey())) {
                    LOGGER.info(String.format(
                        "  %-26s        0 / %,8d (clean)",
                        e.getKey(), e.getValue().get()));
                }
            }
        }
    }

    public static void reset() {
        gpuCalls.set(0);
        gpuTotalNanos.set(0);
        gpuTotalPoints.set(0);
        cpuCalls.set(0);
        cpuTotalNanos.set(0);
        batchFlushes.set(0);
        batchTotalPoints.set(0);
        synchronized (batchLock) { largestBatch = 0; }
        mismatches.clear();
        samplesChecked.clear();
        perTypeStats.clear();
        synchronized (GlassPaperBenchmark.class) { maxMismatch = 0; }
        LOGGER.info("Benchmark counters reset.");
    }

    private static final java.util.concurrent.ConcurrentHashMap<String, AtomicLong> functionTypes
        = new java.util.concurrent.ConcurrentHashMap<>();

    public static void recordFunctionType(String type) {
        functionTypes.computeIfAbsent(type, k -> new AtomicLong()).incrementAndGet();
    }

    public static void recordBatchFlush(int points) {
        batchFlushes.incrementAndGet();
        batchTotalPoints.addAndGet(points);
        synchronized (batchLock) {
            if (points > largestBatch) largestBatch = points;
        }
    }

    private static final java.util.concurrent.ConcurrentHashMap<String, AtomicLong>
        compileFailures = new java.util.concurrent.ConcurrentHashMap<>();

    public static void recordCompileFailure(String reason) {
        compileFailures.computeIfAbsent(reason, k -> new AtomicLong()).incrementAndGet();
    }

}
