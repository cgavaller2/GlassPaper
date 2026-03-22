package io.papermc.paper.gpu;

import net.minecraft.util.RandomSource;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;

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
    // Should be effectively zero — any delta indicates a bug.
    private static final double EPSILON = 1e-9;

    public static boolean validate(GpuNoiseKernel kernel) {
        LOGGER.info("[GlassPaper] Running GPU noise validation (" + SAMPLE_COUNT + " samples)...");

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
            LOGGER.severe("[GlassPaper] Validation failed — could not reflect ImprovedNoise: " + e.getMessage());
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
                        "[GlassPaper] MISMATCH at sample %d: pos=(%.4f,%.4f,%.4f) cpu=%.15f gpu=%.15f delta=%.2e",
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
                "[GlassPaper] Validation PASSED. %d samples, max delta = %.2e",
                SAMPLE_COUNT, maxDelta
            ));
            return true;
        } else {
            LOGGER.severe(String.format(
                "[GlassPaper] Validation FAILED. %d/%d samples mismatched, max delta = %.2e",
                failures, SAMPLE_COUNT, maxDelta
            ));
            return false;
        }
    }

    /**
     * Validates the full NormalNoise GPU path against CPU reference.
     */
    public static boolean validateNormalNoise(GpuNoiseKernel kernel) {
        LOGGER.info("[GlassPaper] Running NormalNoise GPU validation (" + SAMPLE_COUNT + " samples)...");

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
                            "[GlassPaper] NormalNoise MISMATCH at sample %d: cpu=%.15f gpu=%.15f delta=%.2e",
                            i, cpuResults[i], gpuResults[i], delta));
                    }
                }
            }

            if (failures == 0) {
                LOGGER.info(String.format(
                    "[GlassPaper] NormalNoise validation PASSED. %d samples, max delta = %.2e",
                    SAMPLE_COUNT, maxDelta));
                return true;
            } else {
                LOGGER.severe(String.format(
                    "[GlassPaper] NormalNoise validation FAILED. %d/%d mismatched, max delta = %.2e",
                    failures, SAMPLE_COUNT, maxDelta));
                return false;
            }
        } catch (Exception e) {
            LOGGER.severe("[GlassPaper] NormalNoise validation exception: " + e.getMessage());
            return false;
        }
    }

}
