package io.papermc.paper.gpu;

import java.util.List;
import java.util.logging.Logger;

/**
 * Phase 9.14 — fuses N already-compiled density functions into a single
 * {@link FusedDensityFunction} suitable for one-dispatch GPU evaluation
 * via the evalDensityTreeFused kernel.
 *
 * In a separate class from DensityFunctionCompiler (which is in
 * src/minecraft/, the paperweight-patched vanilla code) so this
 * change lives entirely in the outer-repo gpu package — no patch
 * rebuild required.
 *
 * The fusion process:
 *   1. Compute per-kernel global offsets in each shared buffer space
 *      (noise index, octave index, spline header index, splineFloatPool
 *      offset, splineChildren offset, blended index).
 *   2. Allocate the flat arrays sized by the totals.
 *   3. For each constituent CDF:
 *        a. Copy bytecode (iOps, dArgs) into the flat arrays.
 *        b. Walk the bytecode and bump every opcode that carries a
 *           static-buffer index (NOISE/SHIFTED_NOISE/SHIFT_B_NOISE/
 *           WEIRD_SCALED_SAMPLER → +noiseOff; SPLINE_EVAL → +splineOff;
 *           BLENDED_NOISE → +blendedOff). STORE/LOAD_SCRATCH are NOT
 *           re-indexed (scratch is per-work-item, not shared).
 *        c. Copy static buffers, bumping internal references:
 *           - noiseInfo[i].firstOctaveOffset/secondOctaveOffset += octaveOff
 *           - splineHeaders[i].fpStart += splineFloatPoolOff
 *           - splineHeaders[i].extra += splineChildrenOff (when type==2)
 *           - splineChildren[i] += splineHeaderOff (each entry is a header idx)
 *           - blendedPerlinInfo[i].minOctOff/maxOctOff/mainOctOff += octaveOff
 *
 * The kernel then reads from the flat buffers with absolute indices —
 * no runtime offset application needed.
 */
public final class DensityFunctionFuser {

    private static final Logger LOGGER = Logger.getLogger("GlassPaper");

    private DensityFunctionFuser() {}

    public static FusedDensityFunction fuse(List<CompiledDensityFunction> cdfs) {
        if (cdfs == null || cdfs.isEmpty()) {
            throw new IllegalArgumentException("Cannot fuse empty list of CDFs");
        }
        int n = cdfs.size();

        // Per-kernel global offsets, length n+1 (last entry = total).
        int[] iOpStart      = new int[n + 1];
        int[] dArgStart     = new int[n + 1];
        int[] noiseStart    = new int[n + 1];  // unit: noise count
        int[] octaveStart   = new int[n + 1];  // unit: octave count
        int[] splineStart   = new int[n + 1];  // unit: spline header count
        int[] splineFpStart = new int[n + 1];  // unit: float pool element count
        int[] splineChStart = new int[n + 1];  // unit: spline children count
        int[] blendedStart  = new int[n + 1];  // unit: blended noise count

        for (int i = 0; i < n; i++) {
            CompiledDensityFunction cdf = cdfs.get(i);
            iOpStart     [i + 1] = iOpStart     [i] + cdf.iOps.length;
            dArgStart    [i + 1] = dArgStart    [i] + cdf.dArgs.length;
            noiseStart   [i + 1] = noiseStart   [i] + cdf.noiseInfo.length / 4;
            octaveStart  [i + 1] = octaveStart  [i] + cdf.octaveParams.length / 4;
            splineStart  [i + 1] = splineStart  [i] + cdf.splineHeaders.length / 4;
            splineFpStart[i + 1] = splineFpStart[i] + cdf.splineFloatPool.length;
            splineChStart[i + 1] = splineChStart[i] + cdf.splineChildren.length;
            blendedStart [i + 1] = blendedStart [i] + cdf.blendedPerlinInfo.length / 6;
        }

        // Allocate flat buffers.
        int[]    flatIOps                 = new int[iOpStart[n]];
        double[] flatDArgs                = new double[dArgStart[n]];
        double[] flatNoiseParams          = new double[noiseStart[n] * 5];
        int[]    flatNoiseInfo            = new int[noiseStart[n] * 4];
        double[] flatOctaveParams         = new double[octaveStart[n] * 4];
        byte[]   flatPermTables           = new byte[octaveStart[n] * 256];
        int[]    flatSplineHeaders        = new int[splineStart[n] * 4];
        float[]  flatSplineFloatPool      = new float[splineFpStart[n]];
        int[]    flatSplineChildren       = new int[splineChStart[n]];
        double[] flatBlendedScalars       = new double[blendedStart[n] * 5];
        double[] flatBlendedPerlinFactors = new double[blendedStart[n] * 6];
        int[]    flatBlendedPerlinInfo    = new int[blendedStart[n] * 6];

        // Per-kernel: copy + re-index.
        for (int k = 0; k < n; k++) {
            CompiledDensityFunction cdf = cdfs.get(k);
            int noiseOff   = noiseStart[k];
            int octaveOff  = octaveStart[k];
            int splineOff  = splineStart[k];
            int spFpOff    = splineFpStart[k];
            int spChOff    = splineChStart[k];
            int blendedOff = blendedStart[k];

            // 1. Copy iOps, then walk + re-index in-place.
            System.arraycopy(cdf.iOps, 0, flatIOps, iOpStart[k], cdf.iOps.length);
            rewriteBytecodeIndices(
                flatIOps, iOpStart[k], iOpStart[k + 1],
                noiseOff, splineOff, blendedOff);

            // 2. Copy dArgs verbatim — they carry no indices into shared buffers.
            System.arraycopy(cdf.dArgs, 0, flatDArgs, dArgStart[k], cdf.dArgs.length);

            // 3. Copy noiseParams verbatim — also no internal indices.
            System.arraycopy(
                cdf.noiseParams, 0, flatNoiseParams, noiseOff * 5, cdf.noiseParams.length);

            // 4. Copy noiseInfo + bump octave offsets in entries [0] and [1].
            //    Layout: [firstOctaveOffset, secondOctaveOffset, firstCount, secondCount].
            System.arraycopy(cdf.noiseInfo, 0, flatNoiseInfo, noiseOff * 4, cdf.noiseInfo.length);
            int numNoises = cdf.noiseInfo.length / 4;
            for (int i = 0; i < numNoises; i++) {
                int base = noiseOff * 4 + i * 4;
                flatNoiseInfo[base + 0] += octaveOff;  // firstOctaveOffset
                flatNoiseInfo[base + 1] += octaveOff;  // secondOctaveOffset
                // [base+2], [base+3] are counts — leave unchanged
            }

            // 5. Copy octaveParams + permTables verbatim.
            System.arraycopy(
                cdf.octaveParams, 0, flatOctaveParams, octaveOff * 4, cdf.octaveParams.length);
            System.arraycopy(
                cdf.permTables, 0, flatPermTables, octaveOff * 256, cdf.permTables.length);

            // 6. Copy splineHeaders + bump fpStart (entry [2]) and extra (entry [3]
            //    when type==2; for type==0 extra is unused, type==1 extra is also
            //    a float-pool offset so bump by spFpOff).
            //    Layout per spline: [type, knotCount, fpStart, extra].
            System.arraycopy(
                cdf.splineHeaders, 0, flatSplineHeaders, splineOff * 4, cdf.splineHeaders.length);
            int numSplines = cdf.splineHeaders.length / 4;
            for (int i = 0; i < numSplines; i++) {
                int base = splineOff * 4 + i * 4;
                int type = flatSplineHeaders[base + 0];
                flatSplineHeaders[base + 2] += spFpOff;  // fpStart
                if (type == 1) {
                    flatSplineHeaders[base + 3] += spFpOff;  // extra = valStart (in floatPool)
                } else if (type == 2) {
                    flatSplineHeaders[base + 3] += spChOff;  // extra = csStart (in children)
                }
                // type==0: extra is unused
            }

            // 7. Copy splineFloatPool verbatim.
            System.arraycopy(
                cdf.splineFloatPool, 0, flatSplineFloatPool, spFpOff, cdf.splineFloatPool.length);

            // 8. Copy splineChildren + bump each entry (= spline header index).
            System.arraycopy(
                cdf.splineChildren, 0, flatSplineChildren, spChOff, cdf.splineChildren.length);
            for (int i = 0; i < cdf.splineChildren.length; i++) {
                flatSplineChildren[spChOff + i] += splineOff;
            }

            // 9. Copy blendedScalars, blendedPerlinFactors verbatim.
            System.arraycopy(
                cdf.blendedScalars, 0, flatBlendedScalars,
                blendedOff * 5, cdf.blendedScalars.length);
            System.arraycopy(
                cdf.blendedPerlinFactors, 0, flatBlendedPerlinFactors,
                blendedOff * 6, cdf.blendedPerlinFactors.length);

            // 10. Copy blendedPerlinInfo + bump octave offsets in entries [0], [1], [2].
            //     Layout: [minOctOff, maxOctOff, mainOctOff, minCnt, maxCnt, mainCnt].
            System.arraycopy(
                cdf.blendedPerlinInfo, 0, flatBlendedPerlinInfo,
                blendedOff * 6, cdf.blendedPerlinInfo.length);
            int numBlended = cdf.blendedPerlinInfo.length / 6;
            for (int i = 0; i < numBlended; i++) {
                int base = blendedOff * 6 + i * 6;
                flatBlendedPerlinInfo[base + 0] += octaveOff;  // minOctOff
                flatBlendedPerlinInfo[base + 1] += octaveOff;  // maxOctOff
                flatBlendedPerlinInfo[base + 2] += octaveOff;  // mainOctOff
                // [base+3..5] are counts — leave unchanged
            }
        }

        return new FusedDensityFunction(
            n,
            iOpStart, dArgStart,
            flatIOps, flatDArgs,
            flatNoiseParams, flatNoiseInfo,
            flatOctaveParams, flatPermTables,
            flatSplineHeaders, flatSplineFloatPool, flatSplineChildren,
            flatBlendedScalars, flatBlendedPerlinFactors, flatBlendedPerlinInfo);
    }

    /**
     * Walk the bytecode in [start, end), bumping every opcode's index field
     * that refers to a shared static buffer. Mirrors the opcode footprint
     * of the kernel's switch statement in density.cl — every opcode that
     * the VM `case`s on must be handled here with the correct ip advance
     * so we land on the next opcode byte and not in the middle of an iArg.
     *
     * STORE_SCRATCH/LOAD_SCRATCH carry an idx but it's a scratch slot
     * (per-work-item, 32 slots, not shared) — DO NOT bump.
     *
     * Y_GRADIENT carries [fromY, toY] iArgs — they're Y-coordinate values,
     * not indices, so DO NOT bump.
     */
    private static void rewriteBytecodeIndices(int[] flatIOps, int start, int end,
                                               int noiseOff, int splineOff, int blendedOff) {
        int ip = start;
        while (ip < end) {
            int op = flatIOps[ip++];
            if (op == DfOpcode.HALT) break;

            switch (op) {
                // Opcodes with NO iArgs (just opcode byte):
                case DfOpcode.PUSH_X:
                case DfOpcode.PUSH_Y:
                case DfOpcode.PUSH_Z:
                case DfOpcode.ADD:
                case DfOpcode.MUL:
                case DfOpcode.MIN_OP:
                case DfOpcode.MAX_OP:
                case DfOpcode.ABS:
                case DfOpcode.SQUARE:
                case DfOpcode.CUBE:
                case DfOpcode.HALF_NEGATIVE:
                case DfOpcode.QUARTER_NEGATIVE:
                case DfOpcode.SQUEEZE:
                case DfOpcode.INVERT:
                case DfOpcode.BLEND_DENSITY_NOOP:
                    break;

                // Opcodes with iArgs that are NOT shared-buffer indices:
                case DfOpcode.PUSH_CONST:
                case DfOpcode.ADD_CONST:
                case DfOpcode.MUL_CONST:
                case DfOpcode.CLAMP:
                case DfOpcode.RANGE_SELECT:
                    // dArgs only, no iArgs to advance past
                    break;

                case DfOpcode.Y_GRADIENT:
                    // iArgs: [fromY, toY] — Y coordinates, NOT indices
                    ip += 2;
                    break;

                case DfOpcode.STORE_SCRATCH:
                case DfOpcode.LOAD_SCRATCH:
                    // iArg: [scratch_idx] — per-work-item scratch, NOT bumped
                    ip += 1;
                    break;

                // Opcodes carrying NOISE indices:
                case DfOpcode.NOISE:
                case DfOpcode.SHIFTED_NOISE:
                case DfOpcode.SHIFT_B_NOISE:
                    flatIOps[ip] += noiseOff;
                    ip += 1;
                    break;

                case DfOpcode.WEIRD_SCALED_SAMPLER:
                    // iArgs: [noise_idx, mapper_type] — only the first is a noise index
                    flatIOps[ip] += noiseOff;
                    ip += 2;
                    break;

                // Opcode carrying SPLINE header index:
                case DfOpcode.SPLINE_EVAL:
                    // iArgs: [spline_idx, depth]
                    flatIOps[ip] += splineOff;
                    ip += 2;
                    break;

                // Opcode carrying BLENDED index:
                case DfOpcode.BLENDED_NOISE:
                    flatIOps[ip] += blendedOff;
                    ip += 1;
                    break;

                default:
                    LOGGER.warning(String.format(
                        "DensityFunctionFuser: unknown opcode %d at ip=%d (start=%d). "
                      + "Bytecode walker may be desynchronized; fused output likely wrong.",
                        op, ip - 1, start));
                    // Best-effort: skip and hope the next byte is an opcode.
                    break;
            }
        }
    }
}
