package io.papermc.paper.gpu;

/**
 * The compiled representation of a density function tree,
 * ready to dispatch to the GPU evalDensityTree kernel.
 */
public final class CompiledDensityFunction {

    // ── Main program ─────────────────────────────────────────────────────────
    public final int[]    iOps;      // integer opcode/arg stream
    public final double[] dArgs;     // double argument stream

    // ── NormalNoise data (one entry per unique NormalNoise instance) ──────────
    // 5 doubles per noise: [valueFactor, firstInputFactor, firstValueFactor,
    //                        secondInputFactor, secondValueFactor]
    public final double[] noiseParams;

    // 4 ints per noise: [firstOctaveOffset, secondOctaveOffset,
    //                    firstOctaveCount,  secondOctaveCount]
    public final int[]    noiseInfo;

    // 4 doubles per octave: [amplitude, xo, yo, zo]
    public final double[] octaveParams;

    // 256 bytes per octave: permutation table
    public final byte[]   permTables;

    // ── Spline data ───────────────────────────────────────────────────────────
    // 4 ints per spline: [type, knotCount, fpStart, extra]
    //   type 0 (constant):            extra = unused, floatPool[fpStart] = value
    //   type 1 (multipoint + floats): extra = valStart in floatPool
    //   type 2 (multipoint + splines):extra = csStart in childIndices
    public final int[]   splineHeaders;
    public final float[] splineFloatPool;  // locations, derivatives, float-values
    public final int[]   splineChildren;   // child spline indices
    // BlendedNoise data
    // blendedScalars: 5 doubles per instance [xzMultiplier, yMultiplier, xzFactor, yFactor, smearScaleMultiplier]
    public final double[] blendedScalars;
    // blendedPerlinFactors: 6 doubles per instance [minInputF, minValueF, maxInputF, maxValueF, mainInputF, mainValueF]
    public final double[] blendedPerlinFactors;
    // blendedPerlinInfo: 6 ints per instance [minOctOff, maxOctOff, mainOctOff, minOctCount, maxOctCount, mainOctCount]
    public final int[]    blendedPerlinInfo;

    public CompiledDensityFunction(
        int[] iOps, double[] dArgs,
        double[] noiseParams, int[] noiseInfo,
        double[] octaveParams, byte[] permTables,
        int[] splineHeaders, float[] splineFloatPool, int[] splineChildren,
        double[] blendedScalars, double[] blendedPerlinFactors, int[] blendedPerlinInfo) {
        this.iOps               = iOps;
        this.dArgs              = dArgs;
        this.noiseParams        = noiseParams;
        this.noiseInfo          = noiseInfo;
        this.octaveParams       = octaveParams;
        this.permTables         = permTables;
        this.splineHeaders      = splineHeaders;
        this.splineFloatPool    = splineFloatPool;
        this.splineChildren     = splineChildren;
        this.blendedScalars     = blendedScalars;
        this.blendedPerlinFactors = blendedPerlinFactors;
        this.blendedPerlinInfo  = blendedPerlinInfo;
    }
}
