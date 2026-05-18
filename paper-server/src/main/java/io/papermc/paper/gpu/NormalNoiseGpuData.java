package io.papermc.paper.gpu;

import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import net.minecraft.world.level.levelgen.DensityFunction;

import java.lang.reflect.Field;

/**
 * Extracts NormalNoise internals via reflection and packs them into
 * flat arrays suitable for GPU dispatch.
 *
 * Buffer layout matches sampleNormalNoise kernel in density.cl exactly.
 */
public final class NormalNoiseGpuData {

    // noiseParams: [valueFactor, first.inputFactor, first.valueFactor,
    //               second.inputFactor, second.valueFactor]
    public final double[] noiseParams;

    // octaveParams: per octave [amplitude, xo, yo, zo]
    //               first perlin octaves, then second perlin octaves
    public final double[] octaveParams;

    // permTables: 256 bytes per octave, same ordering as octaveParams
    public final byte[] permTables;

    public final int firstNumOctaves;
    public final int secondNumOctaves;

    private NormalNoiseGpuData(double[] noiseParams, double[] octaveParams,
                               byte[] permTables, int firstNumOctaves, int secondNumOctaves) {
        this.noiseParams      = noiseParams;
        this.octaveParams     = octaveParams;
        this.permTables       = permTables;
        this.firstNumOctaves  = firstNumOctaves;
        this.secondNumOctaves = secondNumOctaves;
    }

    // ── Reflection field handles ─────────────────────────────────────────────
    // Cached so we only pay the reflection cost once per server startup.

    private static Field normalNoise_first;
    private static Field normalNoise_second;
    private static Field normalNoise_valueFactor;
    private static Field perlinNoise_noiseLevels;
    private static Field perlinNoise_lowestFreqInputFactor;
    private static Field perlinNoise_lowestFreqValueFactor;
    private static Field perlinNoise_amplitudes;
    private static Field improvedNoise_p;
    private static Field improvedNoise_xo;
    private static Field improvedNoise_yo;
    private static Field improvedNoise_zo;
    private static Field noiseHolder_noise;

    static {
        try {
            normalNoise_first             = field(NormalNoise.class, "first");
            normalNoise_second            = field(NormalNoise.class, "second");
            normalNoise_valueFactor       = field(NormalNoise.class, "valueFactor");
            perlinNoise_noiseLevels       = field(PerlinNoise.class, "noiseLevels");
            perlinNoise_lowestFreqInputFactor = field(PerlinNoise.class, "lowestFreqInputFactor");
            perlinNoise_lowestFreqValueFactor = field(PerlinNoise.class, "lowestFreqValueFactor");
            perlinNoise_amplitudes        = field(PerlinNoise.class, "amplitudes");
            improvedNoise_p               = field(ImprovedNoise.class, "p");
            improvedNoise_xo              = field(ImprovedNoise.class, "xo");
            improvedNoise_yo              = field(ImprovedNoise.class, "yo");
            improvedNoise_zo              = field(ImprovedNoise.class, "zo");
            noiseHolder_noise             = field(DensityFunction.NoiseHolder.class, "noise");
        } catch (Exception e) {
            throw new RuntimeException("Failed to cache NormalNoise reflection fields", e);
        }
    }

    private static Field field(Class<?> cls, String name) throws NoSuchFieldException {
        Field f = cls.getDeclaredField(name);
        f.setAccessible(true);
        return f;
    }

    // ── Extraction ───────────────────────────────────────────────────────────

    public static NormalNoiseGpuData extract(NormalNoise normalNoise) throws Exception {
        // 1. Extract top-level fields
        PerlinNoise first       = (PerlinNoise) normalNoise_first.get(normalNoise);
        PerlinNoise second      = (PerlinNoise) normalNoise_second.get(normalNoise);
        double valueFactor      = (double) normalNoise_valueFactor.get(normalNoise);

        // 2. Extract per-Perlin fields
        PerlinData firstData    = extractPerlin(first);
        PerlinData secondData   = extractPerlin(second);

        int totalOctaves = firstData.numOctaves + secondData.numOctaves;

        // 3. Pack noiseParams
        double[] noiseParams = new double[]{
            valueFactor,
            firstData.lowestFreqInputFactor,
            firstData.lowestFreqValueFactor,
            secondData.lowestFreqInputFactor,
            secondData.lowestFreqValueFactor
        };

        // 4. Pack octaveParams and permTables
        double[] octaveParams = new double[totalOctaves * 4];
        byte[]   permTables   = new byte[totalOctaves * 256];

        packOctaves(firstData,  octaveParams, permTables, 0);
        packOctaves(secondData, octaveParams, permTables, firstData.numOctaves);

        return new NormalNoiseGpuData(
            noiseParams, octaveParams, permTables,
            firstData.numOctaves, secondData.numOctaves
        );
    }

    private static void packOctaves(PerlinData pd,
                                    double[] octaveParams, byte[] permTables,
                                    int octaveOffset) throws Exception {
        for (int i = 0; i < pd.numOctaves; i++) {
            int paramBase = (octaveOffset + i) * 4;
            int permBase  = (octaveOffset + i) * 256;
            ImprovedNoise oct = pd.noiseLevels[i];

            if (oct == null || pd.amplitudes.getDouble(i) == 0.0) {
                // Inactive octave — zero everything, kernel will skip
                octaveParams[paramBase + 0] = 0.0;
                octaveParams[paramBase + 1] = 0.0;
                octaveParams[paramBase + 2] = 0.0;
                octaveParams[paramBase + 3] = 0.0;
                // permTables already zeroed by array initialisation
            } else {
                octaveParams[paramBase + 0] = pd.amplitudes.getDouble(i);
                octaveParams[paramBase + 1] = (double) improvedNoise_xo.get(oct);
                octaveParams[paramBase + 2] = (double) improvedNoise_yo.get(oct);
                octaveParams[paramBase + 3] = (double) improvedNoise_zo.get(oct);

                byte[] p = (byte[]) improvedNoise_p.get(oct);
                System.arraycopy(p, 0, permTables, permBase, 256);
            }
        }
    }

    // ── Internal helper ──────────────────────────────────────────────────────

    private record PerlinData(
        ImprovedNoise[] noiseLevels,
        DoubleList amplitudes,
        double lowestFreqInputFactor,
        double lowestFreqValueFactor,
        int numOctaves
    ) {}

    private static PerlinData extractPerlin(PerlinNoise perlin) throws Exception {
        ImprovedNoise[] levels = (ImprovedNoise[]) perlinNoise_noiseLevels.get(perlin);
        DoubleList amplitudes  = (DoubleList) perlinNoise_amplitudes.get(perlin);
        double inputFactor     = (double) perlinNoise_lowestFreqInputFactor.get(perlin);
        double valueFactor     = (double) perlinNoise_lowestFreqValueFactor.get(perlin);
        return new PerlinData(levels, amplitudes, inputFactor, valueFactor, levels.length);
    }

    /**
     * Extracts NormalNoise from a NoiseHolder via reflection.
     * Returns null if the noise hasn't been initialized yet.
     */
    public static NormalNoise extractNoise(DensityFunction.NoiseHolder holder) throws Exception {
        return (NormalNoise) noiseHolder_noise.get(holder);
    }

}
