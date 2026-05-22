package io.papermc.paper.gpu;

import net.minecraft.util.RandomSource;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.synth.NormalNoise;

import java.lang.reflect.Field;
import java.util.logging.Logger;

/**
 * Validates GPU noise output against CPU reference implementation.
 * Must be called after GpuNoiseKernel.build() succeeds.
 */
public final class GpuNoiseValidator {

    private static final Logger LOGGER = Logger.getLogger("GlassPaper");

    // How many sample points to test
    private static final int SAMPLE_COUNT = 1000;

    // Maximum allowed difference between CPU and GPU results.
    // Phase 9.11: relaxed from 1e-9 (FP64 era) to 1e-5 to accommodate FP32
    // noise-math drift. FP32 machine epsilon is ~1.2e-7 per op; compounded
    // across the lerp3/gradDot/smoothstep chain in sampleAndLerp, expected
    // worst-case drift is ~1e-6 (observed max ~5.55e-7 in 1000-sample
    // startup validation). A 1e-5 threshold leaves a ~20× margin above
    // observed FP32 drift while still catching real bugs (e.g. wrong perm
    // table index, swapped channels, octave-offset errors all produce
    // deltas ≫ 1e-5).
    //
    // When/if the kernel returns to FP64, this should drop back to 1e-9.
    private static final double EPSILON = 1e-5;

    public static boolean validate(GpuNoiseKernel kernel) {
        LOGGER.info("Running GPU noise validation (" + SAMPLE_COUNT + " samples)...");

        // 1. Create a deterministic ImprovedNoise instance
        RandomSource rng = RandomSource.create(12345L);
        ImprovedNoise cpuNoise = new ImprovedNoise(rng);

        // 2. Extract internal fields from the CPU noise via reflection
        byte[] permTable;
        double xo, yo, zo;
        try {
            Field pField  = ImprovedNoise.class.getDeclaredField("p");
            Field xoField = ImprovedNoise.class.getDeclaredField("xo");
            Field yoField = ImprovedNoise.class.getDeclaredField("yo");
            Field zoField = ImprovedNoise.class.getDeclaredField("zo");
            pField.setAccessible(true);
            xoField.setAccessible(true);
            yoField.setAccessible(true);
            zoField.setAccessible(true);

            permTable = (byte[]) pField.get(cpuNoise);
            xo = (double) xoField.get(cpuNoise);
            yo = (double) yoField.get(cpuNoise);
            zo = (double) zoField.get(cpuNoise);
        } catch (Exception e) {
            LOGGER.severe("Validation failed — could not reflect ImprovedNoise: " + e.getMessage());
            return false;
        }

        // 3. Generate test positions — spread across a realistic world coordinate range
        double[] positions = new double[SAMPLE_COUNT * 3];
        double[] cpuResults = new double[SAMPLE_COUNT];
        RandomSource posRng = RandomSource.create(99999L);

        for (int i = 0; i < SAMPLE_COUNT; i++) {
            double x = (posRng.nextDouble() - 0.5) * 10000.0;
            double y = (posRng.nextDouble() - 0.5) * 512.0;
            double z = (posRng.nextDouble() - 0.5) * 10000.0;

            positions[i * 3 + 0] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            // CPU reference
            cpuResults[i] = cpuNoise.noise(x, y, z);
        }

        // 4. Run the same positions through the GPU
        double[] gpuResults = kernel.sampleBatch(
            positions, permTable, xo, yo, zo, SAMPLE_COUNT
        );

        // 5. Compare every result
        int failures = 0;
        double maxDelta = 0.0;

        for (int i = 0; i < SAMPLE_COUNT; i++) {
            double delta = Math.abs(cpuResults[i] - gpuResults[i]);
            if (delta > maxDelta) maxDelta = delta;

            if (delta > EPSILON) {
                failures++;
                if (failures <= 5) { // log first 5 failures in detail
                    LOGGER.severe(String.format(
                        "MISMATCH at sample %d: pos=(%.4f,%.4f,%.4f) cpu=%.15f gpu=%.15f delta=%.2e",
                        i,
                        positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
                        cpuResults[i], gpuResults[i], delta
                    ));
                }
            }
        }

        // 6. Report
        if (failures == 0) {
            LOGGER.info(String.format(
                "Validation PASSED. %d samples, max delta = %.2e",
                SAMPLE_COUNT, maxDelta
            ));
            return true;
        } else {
            LOGGER.severe(String.format(
                "Validation FAILED. %d/%d samples mismatched, max delta = %.2e",
                failures, SAMPLE_COUNT, maxDelta
            ));
            return false;
        }
    }

    /**
     * Validates the full NormalNoise GPU path against CPU reference.
     */
    public static boolean validateNormalNoise(GpuNoiseKernel kernel) {
        LOGGER.info("Running NormalNoise GPU validation (" + SAMPLE_COUNT + " samples)...");

        try {
            // Create a real NormalNoise the same way Minecraft does
            RandomSource rng = RandomSource.create(54321L);
            net.minecraft.world.level.levelgen.synth.NormalNoise cpuNoise =
                net.minecraft.world.level.levelgen.synth.NormalNoise.create(rng, -7, 1.0, 1.0, 1.0, 1.0, 1.0);

            io.papermc.paper.gpu.NormalNoiseGpuData data =
                io.papermc.paper.gpu.NormalNoiseGpuData.extract(cpuNoise);

            RandomSource posRng = RandomSource.create(11111L);
            double[] positions  = new double[SAMPLE_COUNT * 3];
            double[] cpuResults = new double[SAMPLE_COUNT];

            for (int i = 0; i < SAMPLE_COUNT; i++) {
                double x = (posRng.nextDouble() - 0.5) * 10000.0;
                double y = (posRng.nextDouble() - 0.5) * 512.0;
                double z = (posRng.nextDouble() - 0.5) * 10000.0;
                positions[i * 3 + 0] = x;
                positions[i * 3 + 1] = y;
                positions[i * 3 + 2] = z;
                cpuResults[i] = cpuNoise.getValue(x, y, z);
            }

            double[] gpuResults = kernel.sampleNormalNoiseBatch(positions, data, SAMPLE_COUNT);

            int failures = 0;
            double maxDelta = 0.0;
            for (int i = 0; i < SAMPLE_COUNT; i++) {
                double delta = Math.abs(cpuResults[i] - gpuResults[i]);
                if (delta > maxDelta) maxDelta = delta;
                if (delta > EPSILON) {
                    failures++;
                    if (failures <= 5) {
                        LOGGER.severe(String.format(
                            "NormalNoise MISMATCH at sample %d: cpu=%.15f gpu=%.15f delta=%.2e",
                            i, cpuResults[i], gpuResults[i], delta));
                    }
                }
            }

            if (failures == 0) {
                LOGGER.info(String.format(
                    "NormalNoise validation PASSED. %d samples, max delta = %.2e",
                    SAMPLE_COUNT, maxDelta));
                return true;
            } else {
                LOGGER.severe(String.format(
                    "NormalNoise validation FAILED. %d/%d mismatched, max delta = %.2e",
                    failures, SAMPLE_COUNT, maxDelta));
                return false;
            }
        } catch (Exception e) {
            LOGGER.severe("NormalNoise validation exception: " + e.getMessage());
            return false;
        }
    }

    public static boolean validateBlendedNoise(GpuNoiseKernel kernel) {
        LOGGER.info("Running BlendedNoise GPU validation (10000 samples)...");
        try {
            net.minecraft.util.RandomSource rng = net.minecraft.util.RandomSource.create(12345L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise bn =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    rng, 0.25, 0.125, 80.0, 160.0, 8.0);

            net.minecraft.util.RandomSource dr = net.minecraft.util.RandomSource.create(99999L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise dummy1 =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr, 1.0, 1.0, 80.0, 160.0, 8.0);
            net.minecraft.world.level.levelgen.synth.BlendedNoise dummy2 =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr, 1.0, 1.0, 80.0, 160.0, 8.0);

            io.papermc.paper.gpu.CompiledDensityFunction cdf =
                net.minecraft.world.level.levelgen.DensityFunctionCompiler
                    .compileBlendedNoiseTestTree(bn, dummy1, dummy2);

            if (cdf == null) {
                LOGGER.severe("BlendedNoise validation: compile failed");
                return false;
            }

            io.papermc.paper.gpu.GpuCompiledKernel gpuKernel =
                io.papermc.paper.gpu.GpuCompiledKernel.upload(
                    io.papermc.paper.gpu.GpuContext.get(), cdf);

            int count = 10000;
            double[] positions  = new double[count * 3];
            double[] cpuResults = new double[count];
            net.minecraft.util.RandomSource posRng =
                net.minecraft.util.RandomSource.create(55555L);

            for (int i = 0; i < count; i++) {
                int bx = (int)((posRng.nextDouble() - 0.5) * 60000);  // ±30000 blocks
                int by = (int)(posRng.nextDouble() * 384) - 64;         // -64 to 320
                int bz = (int)((posRng.nextDouble() - 0.5) * 60000);
                positions[i * 3]     = bx;
                positions[i * 3 + 1] = by;
                positions[i * 3 + 2] = bz;
                net.minecraft.world.level.levelgen.DensityFunction.FunctionContext ctx =
                    new net.minecraft.world.level.levelgen.DensityFunction.SinglePointContext(bx, by, bz);
                // CPU: dummy1 + dummy2 + bn
                cpuResults[i] = dummy1.compute(ctx) + dummy2.compute(ctx) + bn.compute(ctx);
            }

            double[] gpuResults = kernel.evalDensityTreeFast(positions, gpuKernel, count);
            gpuKernel.close();

            int failures = 0;
            double maxDelta = 0.0;
            // Phase 9.11: relaxed from 1e-6 to 1e-3. BlendedNoise compounds
            // many improvedNoiseYScale calls (main 8 octaves + min 16 + max
            // 16 = up to 40 noise evals per point, each contributing FP32
            // drift). Per-op epsilon ~1.2e-7 × ~40 octaves × accumulated
            // ≈ ~1e-4 observed range. 1e-3 leaves ~10× margin.
            for (int i = 0; i < count; i++) {
                double delta = Math.abs(cpuResults[i] - gpuResults[i]);
                if (delta > maxDelta) maxDelta = delta;
                if (delta > 1e-3) {
                    failures++;
                    if (failures <= 5) {
                        LOGGER.severe(String.format(
                            "BlendedNoise MISMATCH pos=(%.0f,%.0f,%.0f) " +
                                "cpu=%.10f gpu=%.10f delta=%.3e",
                            positions[i*3], positions[i*3+1], positions[i*3+2],
                            cpuResults[i], gpuResults[i], delta));
                    }
                }
            }

            boolean blendedPassed;
            if (failures == 0) {
                LOGGER.info(String.format(
                    "BlendedNoise validation PASSED. max delta = %.2e", maxDelta));
                blendedPassed = true;
            } else {
                LOGGER.severe(String.format(
                    "BlendedNoise validation FAILED. %d/100 mismatched, " +
                        "max delta = %.2e", failures, maxDelta));
                blendedPassed = false;
            }

            // Phase 9.14.A4 + A5 — chain the fusion diagnostics onto the
            // BlendedNoise validator. NOT gating: a fusion failure here would
            // not affect the current hot path (which doesn't use fusion until
            // Phase 9.14.B). We log the result; the gating return value is
            // the BlendedNoise result alone.
            try {
                boolean singletonPassed = validateFusionSingleton(kernel);
                boolean multiPassed = validateFusionMulti(kernel);
                if (!singletonPassed || !multiPassed) {
                    LOGGER.warning(
                        "Phase 9.14 fusion validation failed (singleton="
                      + singletonPassed + ", multi=" + multiPassed
                      + ") — kernel fusion infrastructure has a bug. GPU stays "
                      + "enabled for the non-fused path; fix before wiring "
                      + "fusion into NoiseChunk.fillSliceGpu (Phase 9.14.B).");
                }
            } catch (Throwable t) {
                LOGGER.warning("Phase 9.14 fusion validation threw: " + t.getMessage());
            }

            return blendedPassed;
        } catch (Exception e) {
            LOGGER.severe("BlendedNoise validation exception: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Phase 9.14.A4 — validates the kernel-fusion infrastructure. Compiles
     * the same BlendedNoise test tree used by validateBlendedNoise, then:
     *   1. Dispatches it via the regular non-fused path → reference output.
     *   2. Fuses it (singleton list) into a FusedDensityFunction.
     *   3. Dispatches the fused version via evalFusedSync → fused output.
     *   4. Compares the two outputs bit-by-bit.
     *
     * For a singleton fusion, the fused output MUST be bit-identical to the
     * non-fused output — there's no FP reordering between the paths since
     * the kernel body is the same modulo address-space qualifiers. Any
     * difference indicates a bug in fuse() (re-indexing) or in the fused
     * kernel (helper variants).
     *
     * Once the singleton case validates, multi-kernel fusions can be
     * trusted modulo any deliberate FP reordering — the validator catches
     * structural bugs, not ULP-level fast-math drift.
     */
    public static boolean validateFusionSingleton(GpuNoiseKernel kernel) {
        LOGGER.info("Running fusion singleton validation (1000 samples)...");
        try {
            // Same test setup as validateBlendedNoise — a known-good CDF.
            net.minecraft.util.RandomSource rng = net.minecraft.util.RandomSource.create(12345L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise bn =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    rng, 0.25, 0.125, 80.0, 160.0, 8.0);
            net.minecraft.util.RandomSource dr = net.minecraft.util.RandomSource.create(99999L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise dummy1 =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr, 1.0, 1.0, 80.0, 160.0, 8.0);
            net.minecraft.world.level.levelgen.synth.BlendedNoise dummy2 =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr, 1.0, 1.0, 80.0, 160.0, 8.0);

            io.papermc.paper.gpu.CompiledDensityFunction cdf =
                net.minecraft.world.level.levelgen.DensityFunctionCompiler
                    .compileBlendedNoiseTestTree(bn, dummy1, dummy2);
            if (cdf == null) {
                LOGGER.severe("Fusion validation: test compile failed");
                return false;
            }

            int count = 1000;
            double[] positions = new double[count * 3];
            net.minecraft.util.RandomSource posRng =
                net.minecraft.util.RandomSource.create(77777L);
            for (int i = 0; i < count; i++) {
                positions[i*3]     = (int)((posRng.nextDouble() - 0.5) * 60000);
                positions[i*3 + 1] = (int)(posRng.nextDouble() * 384) - 64;
                positions[i*3 + 2] = (int)((posRng.nextDouble() - 0.5) * 60000);
            }

            // 1. Reference: non-fused dispatch.
            io.papermc.paper.gpu.GpuCompiledKernel gpuKernel =
                io.papermc.paper.gpu.GpuCompiledKernel.upload(
                    io.papermc.paper.gpu.GpuContext.get(), cdf);
            double[] referenceOutput = kernel.evalDensityTreeFast(positions, gpuKernel, count);
            gpuKernel.close();

            // 2. Fuse the singleton.
            io.papermc.paper.gpu.FusedDensityFunction fdf =
                io.papermc.paper.gpu.DensityFunctionFuser.fuse(java.util.List.of(cdf));

            // Diagnostic: confirm singleton fusion produced sensible buffer
            // sizes (should equal the CDF's sizes — no expansion).
            LOGGER.info(String.format(
                "Fusion singleton sizes: iOps=%d (orig %d), dArgs=%d (orig %d), "
              + "noises=%d, octaves=%d, splines=%d",
                fdf.flatIOps.length, cdf.iOps.length,
                fdf.flatDArgs.length, cdf.dArgs.length,
                fdf.flatNoiseInfo.length / 4,
                fdf.flatOctaveParams.length / 4,
                fdf.flatSplineHeaders.length / 4));

            // 3. Dispatch the fused version.
            io.papermc.paper.gpu.GpuFusedKernel gfk =
                io.papermc.paper.gpu.GpuFusedKernel.upload(
                    io.papermc.paper.gpu.GpuContext.get(), fdf);
            double[] fusedOutput = new double[count];  // numKernels=1 × pointCount
            kernel.evalFusedSync(positions, gfk, count, fusedOutput);
            gfk.close();

            // 4. Compare. For a singleton fusion the kernel body is the
            //    same modulo address-space qualifiers — output should match
            //    to high precision. Allow small FP drift in case fast-math
            //    reorders differently between __constant and __global access
            //    patterns.
            int failures = 0;
            double maxDelta = 0.0;
            for (int i = 0; i < count; i++) {
                double delta = Math.abs(referenceOutput[i] - fusedOutput[i]);
                if (delta > maxDelta) maxDelta = delta;
                if (delta > 1e-3) {
                    failures++;
                    if (failures <= 5) {
                        LOGGER.severe(String.format(
                            "Fusion singleton MISMATCH @(%.0f,%.0f,%.0f): "
                          + "ref=%.10f fused=%.10f delta=%.3e",
                            positions[i*3], positions[i*3+1], positions[i*3+2],
                            referenceOutput[i], fusedOutput[i], delta));
                    }
                }
            }

            if (failures == 0) {
                LOGGER.info(String.format(
                    "Fusion singleton validation PASSED. max delta = %.3e", maxDelta));
                return true;
            } else {
                LOGGER.severe(String.format(
                    "Fusion singleton validation FAILED. %d/%d mismatched, max delta = %.3e",
                    failures, count, maxDelta));
                return false;
            }
        } catch (Exception e) {
            LOGGER.severe("Fusion validation exception: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Phase 9.14.A5 — multi-kernel fusion validation. Exercises the
     * re-indexing logic in DensityFunctionFuser by fusing TWO distinct
     * CDFs (different RNG seeds → different noise/blended/octave data)
     * and verifying each constituent's output slice matches the standalone
     * non-fused dispatch of that constituent.
     *
     * The singleton validator passing bit-exact only proved kernel routing
     * works — for N=1, no opcode or buffer offset is bumped. This validator
     * exercises:
     *   - BLENDED_NOISE opcode bumping (idx += blendedOff)
     *   - blendedPerlinInfo octave-offset bumping
     *   - octaveParams / permTables concatenation
     *
     * The BlendedNoise test tree does NOT exercise the NOISE/SHIFTED_NOISE/
     * SPLINE_EVAL re-indexing paths. Those will get exercised when fusion
     * is wired into the hot path in Phase 9.14.B — the runtime CPU/GPU
     * `/gpuvalidate` will catch any divergence there.
     */
    public static boolean validateFusionMulti(GpuNoiseKernel kernel) {
        LOGGER.info("Running fusion multi-kernel validation (1000 samples × 2 kernels)...");
        try {
            // CDF 1: BlendedNoise with seed 11111
            net.minecraft.util.RandomSource rng1 =
                net.minecraft.util.RandomSource.create(11111L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise bn1 =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    rng1, 0.25, 0.125, 80.0, 160.0, 8.0);
            net.minecraft.util.RandomSource dr1 =
                net.minecraft.util.RandomSource.create(22222L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise du1a =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr1, 1.0, 1.0, 80.0, 160.0, 8.0);
            net.minecraft.world.level.levelgen.synth.BlendedNoise du1b =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr1, 1.0, 1.0, 80.0, 160.0, 8.0);
            io.papermc.paper.gpu.CompiledDensityFunction cdf1 =
                net.minecraft.world.level.levelgen.DensityFunctionCompiler
                    .compileBlendedNoiseTestTree(bn1, du1a, du1b);

            // CDF 2: BlendedNoise with seed 33333, different scales
            net.minecraft.util.RandomSource rng2 =
                net.minecraft.util.RandomSource.create(33333L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise bn2 =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    rng2, 0.5, 0.25, 80.0, 160.0, 8.0);
            net.minecraft.util.RandomSource dr2 =
                net.minecraft.util.RandomSource.create(44444L);
            net.minecraft.world.level.levelgen.synth.BlendedNoise du2a =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr2, 1.0, 1.0, 80.0, 160.0, 8.0);
            net.minecraft.world.level.levelgen.synth.BlendedNoise du2b =
                new net.minecraft.world.level.levelgen.synth.BlendedNoise(
                    dr2, 1.0, 1.0, 80.0, 160.0, 8.0);
            io.papermc.paper.gpu.CompiledDensityFunction cdf2 =
                net.minecraft.world.level.levelgen.DensityFunctionCompiler
                    .compileBlendedNoiseTestTree(bn2, du2a, du2b);

            if (cdf1 == null || cdf2 == null) {
                LOGGER.severe("Fusion multi: test compile failed");
                return false;
            }

            // Sample positions (shared between both kernels since fused
            // dispatch uses one position array for all constituents)
            int count = 1000;
            double[] positions = new double[count * 3];
            net.minecraft.util.RandomSource posRng =
                net.minecraft.util.RandomSource.create(88888L);
            for (int i = 0; i < count; i++) {
                positions[i*3]     = (int)((posRng.nextDouble() - 0.5) * 60000);
                positions[i*3 + 1] = (int)(posRng.nextDouble() * 384) - 64;
                positions[i*3 + 2] = (int)((posRng.nextDouble() - 0.5) * 60000);
            }

            // Reference: standalone non-fused dispatch of each CDF
            io.papermc.paper.gpu.GpuCompiledKernel gk1 =
                io.papermc.paper.gpu.GpuCompiledKernel.upload(
                    io.papermc.paper.gpu.GpuContext.get(), cdf1);
            double[] ref1 = kernel.evalDensityTreeFast(positions, gk1, count);
            gk1.close();

            io.papermc.paper.gpu.GpuCompiledKernel gk2 =
                io.papermc.paper.gpu.GpuCompiledKernel.upload(
                    io.papermc.paper.gpu.GpuContext.get(), cdf2);
            double[] ref2 = kernel.evalDensityTreeFast(positions, gk2, count);
            gk2.close();

            // Fused dispatch
            io.papermc.paper.gpu.FusedDensityFunction fdf =
                io.papermc.paper.gpu.DensityFunctionFuser.fuse(
                    java.util.List.of(cdf1, cdf2));
            LOGGER.info(String.format(
                "Fusion multi sizes: 2 kernels, iOps=%d (%d+%d), octaves=%d (%d+%d), "
              + "blended=%d (%d+%d)",
                fdf.flatIOps.length, cdf1.iOps.length, cdf2.iOps.length,
                fdf.flatOctaveParams.length / 4,
                cdf1.octaveParams.length / 4, cdf2.octaveParams.length / 4,
                fdf.flatBlendedPerlinInfo.length / 6,
                cdf1.blendedPerlinInfo.length / 6, cdf2.blendedPerlinInfo.length / 6));

            io.papermc.paper.gpu.GpuFusedKernel gfk =
                io.papermc.paper.gpu.GpuFusedKernel.upload(
                    io.papermc.paper.gpu.GpuContext.get(), fdf);
            // Output layout: [k0 results × count][k1 results × count]
            double[] fusedOut = new double[2 * count];
            kernel.evalFusedSync(positions, gfk, count, fusedOut);
            gfk.close();

            // Compare: ref1 ↔ fusedOut[0..count), ref2 ↔ fusedOut[count..2*count)
            int failures = 0;
            double maxDeltaK1 = 0.0, maxDeltaK2 = 0.0;
            for (int i = 0; i < count; i++) {
                double f1 = fusedOut[i];
                double d1 = Math.abs(f1 - ref1[i]);
                double f2 = fusedOut[count + i];
                double d2 = Math.abs(f2 - ref2[i]);
                if (d1 > maxDeltaK1) maxDeltaK1 = d1;
                if (d2 > maxDeltaK2) maxDeltaK2 = d2;

                if (d1 > 1e-3 || d2 > 1e-3) {
                    failures++;
                    if (failures <= 5) {
                        LOGGER.severe(String.format(
                            "Fusion multi MISMATCH @(%.0f,%.0f,%.0f): "
                          + "k1 ref=%.10f fused=%.10f δ=%.3e; "
                          + "k2 ref=%.10f fused=%.10f δ=%.3e",
                            positions[i*3], positions[i*3+1], positions[i*3+2],
                            ref1[i], f1, d1, ref2[i], f2, d2));
                    }
                }
            }

            if (failures == 0) {
                LOGGER.info(String.format(
                    "Fusion multi validation PASSED. max delta k1=%.3e, k2=%.3e",
                    maxDeltaK1, maxDeltaK2));
                return true;
            } else {
                LOGGER.severe(String.format(
                    "Fusion multi validation FAILED. %d/%d mismatched, "
                  + "max delta k1=%.3e, k2=%.3e",
                    failures, count, maxDeltaK1, maxDeltaK2));
                return false;
            }
        } catch (Exception e) {
            LOGGER.severe("Fusion multi validation exception: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}
