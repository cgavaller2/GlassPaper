package io.papermc.paper.gpu;

/**
 * Phase 9.14 — fused representation of N density-function trees, ready
 * to dispatch via the GPU evalDensityTreeFused kernel in a single GPU
 * command instead of N separate evalDensityTree dispatches.
 *
 * Per-chunk-slice we currently submit ~5 distinct density-function
 * kernels (BlendDensity + ~4 RangeChoice variants), each as its own
 * GPU dispatch. Each dispatch carries ~150-850 µs of host overhead
 * (per Phase 9.7/9.13 profiling: kernel exec ~150 µs, host wait ~850
 * µs of which most is driver overhead and dispatch serialization).
 * Fusing those 5 dispatches into 1 amortizes the host overhead 5×
 * — the largest remaining lever after Phase 9.13's async dispatch.
 *
 * Layout: each constituent CompiledDensityFunction's static buffers
 * are concatenated end-to-end. Internal references (e.g., noiseInfo's
 * octave offsets, splineHeaders' fpStart/extra, splineChildren's
 * header indices, blendedPerlinInfo's octave offsets) are bumped by
 * the per-kernel global offset during fusion so the kernel reads
 * them as absolute indices into the flat buffers. Bytecode opcodes
 * carrying static-buffer indices (NOISE/SHIFTED_NOISE/SHIFT_B_NOISE/
 * WEIRD_SCALED_SAMPLER, SPLINE_EVAL, BLENDED_NOISE) are similarly
 * re-indexed. STORE_SCRATCH/LOAD_SCRATCH are NOT re-indexed — scratch
 * is per-work-item, not shared between kernels.
 *
 * Per-kernel iOpStart and dArgStart tell the kernel where each
 * constituent's bytecode + dArgs stream begins in the flat arrays.
 * Each constituent's bytecode is HALT-terminated, so the kernel
 * doesn't need iOpStart[k+1] at runtime — it just reads until HALT.
 * The +1 entry is provided as length sentinel for host-side bookkeeping.
 *
 * Work-item layout for evalDensityTreeFused:
 *   global_id = kernelIdx * pointCount + pointIdx
 *   numWorkItems = numKernels * pointCount
 *   output[kernelIdx * pointCount + pointIdx] = density value
 * The host splits the multi-output back to per-constituent slices.
 */
public final class FusedDensityFunction {

    public final int numKernels;

    // Per-kernel start offsets into flatIOps and flatDArgs, length numKernels+1.
    // The +1 sentinel is the total length (= flat array length).
    public final int[] iOpStart;
    public final int[] dArgStart;

    // ── Concatenated bytecode ────────────────────────────────────────────────
    public final int[]    flatIOps;
    public final double[] flatDArgs;

    // ── Concatenated NormalNoise data ────────────────────────────────────────
    // Sizes follow the same per-noise layout as CompiledDensityFunction:
    //   flatNoiseParams: 5 doubles per noise
    //   flatNoiseInfo:   4 ints per noise (with octave offsets re-indexed)
    public final double[] flatNoiseParams;
    public final int[]    flatNoiseInfo;

    // ── Concatenated octave data ─────────────────────────────────────────────
    //   flatOctaveParams: 4 doubles per octave
    //   flatPermTables:   256 bytes per octave
    public final double[] flatOctaveParams;
    public final byte[]   flatPermTables;

    // ── Concatenated spline data ─────────────────────────────────────────────
    //   flatSplineHeaders:   4 ints per spline (with fpStart/extra re-indexed)
    //   flatSplineFloatPool: floats (knot locations, derivatives, values)
    //   flatSplineChildren:  child spline indices (re-indexed into flat headers)
    public final int[]   flatSplineHeaders;
    public final float[] flatSplineFloatPool;
    public final int[]   flatSplineChildren;

    // ── Concatenated BlendedNoise data ───────────────────────────────────────
    //   flatBlendedScalars:       5 doubles per blended-noise instance
    //   flatBlendedPerlinFactors: 6 doubles per instance
    //   flatBlendedPerlinInfo:    6 ints per instance (octave offsets re-indexed)
    public final double[] flatBlendedScalars;
    public final double[] flatBlendedPerlinFactors;
    public final int[]    flatBlendedPerlinInfo;

    public FusedDensityFunction(
        int numKernels,
        int[] iOpStart, int[] dArgStart,
        int[] flatIOps, double[] flatDArgs,
        double[] flatNoiseParams, int[] flatNoiseInfo,
        double[] flatOctaveParams, byte[] flatPermTables,
        int[] flatSplineHeaders, float[] flatSplineFloatPool, int[] flatSplineChildren,
        double[] flatBlendedScalars, double[] flatBlendedPerlinFactors, int[] flatBlendedPerlinInfo) {
        this.numKernels               = numKernels;
        this.iOpStart                 = iOpStart;
        this.dArgStart                = dArgStart;
        this.flatIOps                 = flatIOps;
        this.flatDArgs                = flatDArgs;
        this.flatNoiseParams          = flatNoiseParams;
        this.flatNoiseInfo            = flatNoiseInfo;
        this.flatOctaveParams         = flatOctaveParams;
        this.flatPermTables           = flatPermTables;
        this.flatSplineHeaders        = flatSplineHeaders;
        this.flatSplineFloatPool      = flatSplineFloatPool;
        this.flatSplineChildren       = flatSplineChildren;
        this.flatBlendedScalars       = flatBlendedScalars;
        this.flatBlendedPerlinFactors = flatBlendedPerlinFactors;
        this.flatBlendedPerlinInfo    = flatBlendedPerlinInfo;
    }
}
