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

    // Compile-pass cost (DensityFunctionCompiler.compile + DfCacheKey.of)
    // Runs on Paper worker threads in GpuKernelHolder.getOrUpload.
    // Split into cache-miss (full compile + upload) and cache-hit
    // (compile + key-lookup only). Identity-cache hits skip both.
    private static final AtomicLong compileMissCalls = new AtomicLong();
    private static final AtomicLong compileMissNanos = new AtomicLong();
    private static final AtomicLong compileHitCalls  = new AtomicLong();
    private static final AtomicLong compileHitNanos  = new AtomicLong();
    private static final AtomicLong identityHitCalls = new AtomicLong();

    // Phase 9.7 — per-phase GPU timing from cl_event timestamps. Captured
    // only when profileMode is on (toggle via `gpuprofile` console command).
    // CL_PROFILING_COMMAND_START / END measures actual on-device time.
    // CL_PROFILING_COMMAND_SUBMIT - QUEUED = host queue overhead.
    // CL_PROFILING_COMMAND_START - SUBMIT = driver scheduling latency.
    private static volatile boolean profileMode = false;
    // Sample 1 in N dispatches when profiling is on. NVIDIA OpenCL appears
    // to lose profile-info tracking under heavy concurrent dispatch (37
    // samples succeeded then CL_PROFILING_INFO_NOT_AVAILABLE started firing
    // — driver-side per-event-tracking limit hypothesis). Sampling spreads
    // the captures and reduces concurrent-event pressure on the driver.
    private static volatile int profileSampleEvery = 32;
    private static final AtomicLong profileDispatchSeen = new AtomicLong();

    private static final AtomicLong profileSamples    = new AtomicLong();
    private static final AtomicLong profileFailures   = new AtomicLong();
    private static final AtomicLong profileWriteNanos = new AtomicLong();
    private static final AtomicLong profileKernelNanos = new AtomicLong();
    private static final AtomicLong profileReadNanos  = new AtomicLong();
    private static final AtomicLong profileQueueLatencyNanos = new AtomicLong();
    private static final AtomicLong profileWallNanos  = new AtomicLong();

    public static boolean isProfiling() { return profileMode; }
    public static void setProfiling(boolean p) { profileMode = p; }
    public static int getProfileSampleEvery() { return profileSampleEvery; }
    public static void setProfileSampleEvery(int n) {
        profileSampleEvery = Math.max(1, n);
    }

    /**
     * Should this particular dispatch capture profiling events?
     * Called once per dispatch when profileMode is on. Returns true 1 in
     * profileSampleEvery times. Atomic counter so concurrent dispatchers
     * don't all sample the same indices.
     */
    public static boolean shouldProfileThisDispatch() {
        if (!profileMode) return false;
        long n = profileDispatchSeen.incrementAndGet();
        return (n % profileSampleEvery) == 0;
    }

    public static void recordProfileFailure() {
        profileFailures.incrementAndGet();
    }

    public static void recordProfile(long writeNs, long kernelNs, long readNs,
                                     long queueLatencyNs, long wallNs) {
        profileSamples.incrementAndGet();
        profileWriteNanos.addAndGet(writeNs);
        profileKernelNanos.addAndGet(kernelNs);
        profileReadNanos.addAndGet(readNs);
        profileQueueLatencyNanos.addAndGet(queueLatencyNs);
        profileWallNanos.addAndGet(wallNs);
    }

    private static final AtomicLong batchFlushes     = new AtomicLong();
    private static final AtomicLong batchTotalPoints  = new AtomicLong();
    private static long             largestBatch      = 0;
    private static final Object     batchLock         = new Object();

    // Phase 9.12.A — per-slot static-arg cache hit/miss. Records whether the
    // 12 GpuCompiledKernel buffer args were already bound from a previous
    // dispatch on the same slot. High hit rate = lower clSetKernelArg cost
    // per dispatch. Diagnostic: if hit rate is low (< 30%), consider pinning
    // kernels to specific slots (kernel-hash-to-slot affinity).
    private static final AtomicLong argCacheHits   = new AtomicLong();
    private static final AtomicLong argCacheMisses = new AtomicLong();
    public static void recordArgCacheHit()   { argCacheHits.incrementAndGet(); }
    public static void recordArgCacheMiss()  { argCacheMisses.incrementAndGet(); }

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

    /** Tracks which density-function class names have had their bytecode dumped. */
    private static final java.util.Set<String> bytecodeDumped =
        java.util.concurrent.ConcurrentHashMap.newKeySet();

    /**
     * Dump the compiled GPU bytecode for this density function once per type
     * (keyed on getClass().getSimpleName()), with each opcode annotated.
     * Called from NoiseChunk.fillSliceGpu on the first mismatch of a given
     * type — gives us the actual bytecode stream to inspect for runtime bugs.
     */
    public static void dumpCompiledBytecode(String typeName,
                                            net.minecraft.world.level.levelgen.DensityFunction fn) {
        if (!bytecodeDumped.add(typeName)) return;
        try {
            io.papermc.paper.gpu.CompiledDensityFunction cdf =
                net.minecraft.world.level.levelgen.DensityFunctionCompiler.compile(fn);
            if (cdf == null) {
                LOGGER.warning("Bytecode dump for " + typeName + ": compile returned null");
                return;
            }

            StringBuilder sb = new StringBuilder();
            sb.append("════ Compiled bytecode for ").append(typeName)
              .append(" (").append(cdf.iOps.length).append(" iOps, ")
              .append(cdf.dArgs.length).append(" dArgs, ")
              .append(cdf.octaveParams.length / 4).append(" octaves, ")
              .append(cdf.blendedScalars.length / 5).append(" blended noises, ")
              .append(cdf.splineHeaders.length / 4).append(" splines) ════\n");

            int ip = 0, dp = 0;
            int step = 0;
            while (ip < cdf.iOps.length && step < 500) {
                int op = cdf.iOps[ip++];
                sb.append(String.format("  [%4d] ", step++));
                switch (op) {
                    case io.papermc.paper.gpu.DfOpcode.PUSH_CONST:
                        sb.append("PUSH_CONST ").append(cdf.dArgs[dp++]); break;
                    case io.papermc.paper.gpu.DfOpcode.PUSH_X: sb.append("PUSH_X");   break;
                    case io.papermc.paper.gpu.DfOpcode.PUSH_Y: sb.append("PUSH_Y");   break;
                    case io.papermc.paper.gpu.DfOpcode.PUSH_Z: sb.append("PUSH_Z");   break;
                    case io.papermc.paper.gpu.DfOpcode.ADD:    sb.append("ADD");      break;
                    case io.papermc.paper.gpu.DfOpcode.MUL:    sb.append("MUL");      break;
                    case io.papermc.paper.gpu.DfOpcode.MIN_OP: sb.append("MIN");      break;
                    case io.papermc.paper.gpu.DfOpcode.MAX_OP: sb.append("MAX");      break;
                    case io.papermc.paper.gpu.DfOpcode.ABS:    sb.append("ABS");      break;
                    case io.papermc.paper.gpu.DfOpcode.SQUARE: sb.append("SQUARE");   break;
                    case io.papermc.paper.gpu.DfOpcode.CUBE:   sb.append("CUBE");     break;
                    case io.papermc.paper.gpu.DfOpcode.HALF_NEGATIVE:    sb.append("HALF_NEG");    break;
                    case io.papermc.paper.gpu.DfOpcode.QUARTER_NEGATIVE: sb.append("QUARTER_NEG"); break;
                    case io.papermc.paper.gpu.DfOpcode.SQUEEZE: sb.append("SQUEEZE"); break;
                    case io.papermc.paper.gpu.DfOpcode.INVERT:  sb.append("INVERT");  break;
                    case io.papermc.paper.gpu.DfOpcode.CLAMP:
                        sb.append(String.format("CLAMP [%g, %g]", cdf.dArgs[dp++], cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.NOISE:
                        sb.append(String.format("NOISE idx=%d xzS=%g yS=%g",
                            cdf.iOps[ip++], cdf.dArgs[dp++], cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.SHIFTED_NOISE:
                        sb.append(String.format("SHIFTED_NOISE idx=%d xzS=%g yS=%g",
                            cdf.iOps[ip++], cdf.dArgs[dp++], cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.SHIFT_B_NOISE:
                        sb.append(String.format("SHIFT_B_NOISE idx=%d xzS=%g",
                            cdf.iOps[ip++], cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.Y_GRADIENT:
                        sb.append(String.format("Y_GRADIENT fy=%d ty=%d fv=%g tv=%g",
                            cdf.iOps[ip++], cdf.iOps[ip++], cdf.dArgs[dp++], cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.RANGE_SELECT:
                        sb.append(String.format("RANGE_SELECT [%g, %g)",
                            cdf.dArgs[dp++], cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.BLEND_DENSITY_NOOP:
                        sb.append("BLEND_DENSITY_NOOP"); break;
                    case io.papermc.paper.gpu.DfOpcode.SPLINE_EVAL:
                        sb.append(String.format("SPLINE_EVAL spline=%d depth=%d",
                            cdf.iOps[ip++], cdf.iOps[ip++])); break;
                    case io.papermc.paper.gpu.DfOpcode.ADD_CONST:
                        sb.append(String.format("ADD_CONST %g", cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.MUL_CONST:
                        sb.append(String.format("MUL_CONST %g", cdf.dArgs[dp++])); break;
                    case io.papermc.paper.gpu.DfOpcode.BLENDED_NOISE:
                        sb.append(String.format("BLENDED_NOISE bn=%d", cdf.iOps[ip++])); break;
                    case io.papermc.paper.gpu.DfOpcode.WEIRD_SCALED_SAMPLER:
                        sb.append(String.format("WEIRD_SCALED_SAMPLER idx=%d mapper=%d",
                            cdf.iOps[ip++], cdf.iOps[ip++])); break;
                    case io.papermc.paper.gpu.DfOpcode.STORE_SCRATCH:
                        sb.append(String.format("STORE_SCRATCH idx=%d", cdf.iOps[ip++])); break;
                    case io.papermc.paper.gpu.DfOpcode.LOAD_SCRATCH:
                        sb.append(String.format("LOAD_SCRATCH idx=%d", cdf.iOps[ip++])); break;
                    case io.papermc.paper.gpu.DfOpcode.HALT:
                        sb.append("HALT");
                        ip = cdf.iOps.length;
                        break;
                    default:
                        sb.append("UNKNOWN op=").append(op);
                        ip = cdf.iOps.length;
                        break;
                }
                sb.append('\n');
            }
            if (step >= 500) sb.append("  ... (truncated, ").append(cdf.iOps.length - ip).append(" bytes remaining)\n");
            LOGGER.warning(sb.toString());
        } catch (Throwable t) {
            LOGGER.warning("Bytecode dump failed: " + t.getMessage());
        }
    }

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

    public static void recordCompileMiss(long nanos) {
        compileMissCalls.incrementAndGet();
        compileMissNanos.addAndGet(nanos);
    }

    public static void recordCompileHit(long nanos) {
        compileHitCalls.incrementAndGet();
        compileHitNanos.addAndGet(nanos);
    }

    public static void recordIdentityHit() {
        identityHitCalls.incrementAndGet();
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

        // Phase 9.7 — GPU profiling breakdown. Tells us where the per-dispatch
        // ms goes: actual kernel exec, PCIe upload, PCIe download, or
        // host/driver scheduling latency. Decides whether further kernel-side
        // optimization (Y-independence, CSE) is worth pursuing vs other paths
        // (CacheOnce priming, batch coalescing tweaks).
        long ps = profileSamples.get();
        long pf = profileFailures.get();
        long pd = profileDispatchSeen.get();
        if (ps > 0 || pf > 0) {
            LOGGER.info(String.format(
                "─── GPU profiling (%,d samples / %,d dispatches seen, 1 in %d sampled, %,d failures) ───",
                ps, pd, profileSampleEvery, pf));
            if (ps > 0) {
                double wMs = profileWriteNanos.get() / 1_000_000.0;
                double kMs = profileKernelNanos.get() / 1_000_000.0;
                double rMs = profileReadNanos.get()  / 1_000_000.0;
                double qMs = profileQueueLatencyNanos.get() / 1_000_000.0;
                double wallMs = profileWallNanos.get() / 1_000_000.0;
                double sumMs = wMs + kMs + rMs;
                LOGGER.info(String.format(
                    "  PCIe upload (write)  : avg %.1f µs   total %8.1f ms   (%.1f%%)",
                    wMs  * 1000.0 / ps, wMs, 100.0 * wMs / wallMs));
                LOGGER.info(String.format(
                    "  Kernel execution     : avg %.1f µs   total %8.1f ms   (%.1f%%)",
                    kMs  * 1000.0 / ps, kMs, 100.0 * kMs / wallMs));
                LOGGER.info(String.format(
                    "  PCIe download (read) : avg %.1f µs   total %8.1f ms   (%.1f%%)",
                    rMs  * 1000.0 / ps, rMs, 100.0 * rMs / wallMs));
                LOGGER.info(String.format(
                    "  Queue / driver gap   : avg %.1f µs   total %8.1f ms   (%.1f%%)",
                    qMs  * 1000.0 / ps, qMs, 100.0 * qMs / wallMs));
                LOGGER.info(String.format(
                    "  Wall time / sample   : avg %.1f µs   (compute %.0f%%, transfer %.0f%%, idle %.0f%%)",
                    wallMs * 1000.0 / ps,
                    100.0 * kMs / sumMs,
                    100.0 * (wMs + rMs) / sumMs,
                    100.0 * (wallMs - sumMs) / wallMs));
            }
        }

        // Phase 9.12.A — static-arg cache hit rate.
        long argHits   = argCacheHits.get();
        long argMisses = argCacheMisses.get();
        if (argHits + argMisses > 0) {
            long argTotal = argHits + argMisses;
            double hitPct = (argHits * 100.0) / argTotal;
            LOGGER.info(String.format(
                "Static-arg cache: %,d hits / %,d total (%.1f%% hit, %,d misses)",
                argHits, argTotal, hitPct, argMisses));
        }

        long cmCalls = compileMissCalls.get(), chCalls = compileHitCalls.get();
        long idCalls = identityHitCalls.get();
        if (cmCalls + chCalls + idCalls > 0) {
            long cmNanos = compileMissNanos.get(), chNanos = compileHitNanos.get();
            long totalLookups = cmCalls + chCalls + idCalls;
            double idPct  = (idCalls * 100.0) / totalLookups;
            double chPct  = (chCalls * 100.0) / totalLookups;
            double cmPct  = (cmCalls * 100.0) / totalLookups;
            LOGGER.info("─── getOrUpload cost (worker thread) ───");
            LOGGER.info(String.format(
                "Identity hits   : %,d  (%.1f%%, ~0 cost)", idCalls, idPct));
            if (chCalls > 0) {
                LOGGER.info(String.format(
                    "Content hits    : %,d  (%.1f%%, %.4f ms avg, %.2f ms total)",
                    chCalls, chPct, (chNanos / (double) chCalls) / 1_000_000.0,
                    chNanos / 1_000_000.0));
            }
            if (cmCalls > 0) {
                LOGGER.info(String.format(
                    "Compile + upload: %,d  (%.1f%%, %.4f ms avg, %.2f ms total)",
                    cmCalls, cmPct, (cmNanos / (double) cmCalls) / 1_000_000.0,
                    cmNanos / 1_000_000.0));
            }
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
        compileMissCalls.set(0);
        compileMissNanos.set(0);
        compileHitCalls.set(0);
        compileHitNanos.set(0);
        identityHitCalls.set(0);
        profileSamples.set(0);
        profileFailures.set(0);
        profileDispatchSeen.set(0);
        profileWriteNanos.set(0);
        profileKernelNanos.set(0);
        profileReadNanos.set(0);
        profileQueueLatencyNanos.set(0);
        profileWallNanos.set(0);
        synchronized (batchLock) { largestBatch = 0; }
        argCacheHits.set(0);
        argCacheMisses.set(0);
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
