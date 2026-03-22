package io.papermc.paper.gpu;

import java.util.Arrays;

/**
 * Cache key based on compiled program content rather than object identity.
 * Two density functions that produce identical iOps share one GPU buffer set.
 */
public record DfCacheKey(int[] iOps, double[] dArgs) {

    public static DfCacheKey of(CompiledDensityFunction cdf) {
        return new DfCacheKey(cdf.iOps, cdf.dArgs);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DfCacheKey other)) return false;
        return Arrays.equals(iOps, other.iOps) && Arrays.equals(dArgs, other.dArgs);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(iOps) * 31 + Arrays.hashCode(dArgs);
    }
}
