package io.papermc.paper.gpu;

import java.util.Arrays;

/**
 * Cache key for compiled density function GPU buffers. Two density functions
 * share one set of GPU buffers iff their full compiled representation is
 * byte-identical — bytecode, inline scalars, noise tables, spline tables,
 * and blended-noise data.
 *
 * Earlier versions of this key hashed only iOps + dArgs, which collided
 * across structurally-identical density trees that referenced different
 * NormalNoise instances (each compile assigns local noise indices 0, 1, 2,
 * so the bytecode `OP_NOISE 0` is identical regardless of which underlying
 * noise that idx 0 refers to). The cached kernel then dispatched with the
 * first-encountered noise's permutation tables for every later collision —
 * producing noise-like-but-wrong values, which manifested as smooth carved
 * artifacts in cave-related density functions.
 */
public record DfCacheKey(
    int[]    iOps,
    double[] dArgs,
    double[] noiseParams,
    int[]    noiseInfo,
    double[] octaveParams,
    byte[]   permTables,
    int[]    splineHeaders,
    float[]  splineFloatPool,
    int[]    splineChildren,
    double[] blendedScalars,
    double[] blendedPerlinFactors,
    int[]    blendedPerlinInfo
) {

    public static DfCacheKey of(CompiledDensityFunction cdf) {
        return new DfCacheKey(
            cdf.iOps, cdf.dArgs,
            cdf.noiseParams, cdf.noiseInfo,
            cdf.octaveParams, cdf.permTables,
            cdf.splineHeaders, cdf.splineFloatPool, cdf.splineChildren,
            cdf.blendedScalars, cdf.blendedPerlinFactors, cdf.blendedPerlinInfo);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DfCacheKey k)) return false;
        return Arrays.equals(iOps,                 k.iOps)
            && Arrays.equals(dArgs,                k.dArgs)
            && Arrays.equals(noiseParams,          k.noiseParams)
            && Arrays.equals(noiseInfo,            k.noiseInfo)
            && Arrays.equals(octaveParams,         k.octaveParams)
            && Arrays.equals(permTables,           k.permTables)
            && Arrays.equals(splineHeaders,        k.splineHeaders)
            && Arrays.equals(splineFloatPool,      k.splineFloatPool)
            && Arrays.equals(splineChildren,       k.splineChildren)
            && Arrays.equals(blendedScalars,       k.blendedScalars)
            && Arrays.equals(blendedPerlinFactors, k.blendedPerlinFactors)
            && Arrays.equals(blendedPerlinInfo,    k.blendedPerlinInfo);
    }

    @Override
    public int hashCode() {
        int h = Arrays.hashCode(iOps);
        h = h * 31 + Arrays.hashCode(dArgs);
        h = h * 31 + Arrays.hashCode(noiseParams);
        h = h * 31 + Arrays.hashCode(noiseInfo);
        h = h * 31 + Arrays.hashCode(octaveParams);
        h = h * 31 + Arrays.hashCode(permTables);
        h = h * 31 + Arrays.hashCode(splineHeaders);
        h = h * 31 + Arrays.hashCode(splineFloatPool);
        h = h * 31 + Arrays.hashCode(splineChildren);
        h = h * 31 + Arrays.hashCode(blendedScalars);
        h = h * 31 + Arrays.hashCode(blendedPerlinFactors);
        h = h * 31 + Arrays.hashCode(blendedPerlinInfo);
        return h;
    }
}
