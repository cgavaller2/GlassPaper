package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;

/**
 * Phase 9.14 — GPU upload of a {@link FusedDensityFunction}. Mirrors
 * {@link GpuCompiledKernel} but for the fused multi-kernel buffer set,
 * plus the two extra per-kernel offset arrays (iOpStart, dArgStart).
 *
 * All buffers are __global on the device (no __constant) because the
 * concatenated sizes may exceed NVIDIA's per-SM constant cache
 * (~8KB on Turing — Phase 9.6.G measurement showed exceeding it caused
 * cache thrashing and a perf regression). For evalDensityTree's small
 * single-kernel inputs we keep __constant; for evalDensityTreeFused's
 * fused inputs we use __global to stay safely cached in L1.
 */
public final class GpuFusedKernel implements AutoCloseable {

    final int numKernels;
    final cl_mem iOpStartBuf;
    final cl_mem dArgStartBuf;
    final cl_mem flatIOpsBuf;
    final cl_mem flatDArgsBuf;
    final cl_mem flatNoiseParamsBuf;
    final cl_mem flatNoiseInfoBuf;
    final cl_mem flatOctaveParamsBuf;
    final cl_mem flatPermTablesBuf;
    final cl_mem flatSplineHeadersBuf;
    final cl_mem flatSplineFloatPoolBuf;
    final cl_mem flatSplineChildrenBuf;
    final cl_mem flatBlendedScalarsBuf;
    final cl_mem flatBlendedPerlinFactorsBuf;
    final cl_mem flatBlendedPerlinInfoBuf;

    private GpuFusedKernel(int numKernels,
            cl_mem iOpStartBuf, cl_mem dArgStartBuf,
            cl_mem flatIOpsBuf, cl_mem flatDArgsBuf,
            cl_mem flatNoiseParamsBuf, cl_mem flatNoiseInfoBuf,
            cl_mem flatOctaveParamsBuf, cl_mem flatPermTablesBuf,
            cl_mem flatSplineHeadersBuf, cl_mem flatSplineFloatPoolBuf, cl_mem flatSplineChildrenBuf,
            cl_mem flatBlendedScalarsBuf, cl_mem flatBlendedPerlinFactorsBuf, cl_mem flatBlendedPerlinInfoBuf) {
        this.numKernels                  = numKernels;
        this.iOpStartBuf                 = iOpStartBuf;
        this.dArgStartBuf                = dArgStartBuf;
        this.flatIOpsBuf                 = flatIOpsBuf;
        this.flatDArgsBuf                = flatDArgsBuf;
        this.flatNoiseParamsBuf          = flatNoiseParamsBuf;
        this.flatNoiseInfoBuf            = flatNoiseInfoBuf;
        this.flatOctaveParamsBuf         = flatOctaveParamsBuf;
        this.flatPermTablesBuf           = flatPermTablesBuf;
        this.flatSplineHeadersBuf        = flatSplineHeadersBuf;
        this.flatSplineFloatPoolBuf      = flatSplineFloatPoolBuf;
        this.flatSplineChildrenBuf       = flatSplineChildrenBuf;
        this.flatBlendedScalarsBuf       = flatBlendedScalarsBuf;
        this.flatBlendedPerlinFactorsBuf = flatBlendedPerlinFactorsBuf;
        this.flatBlendedPerlinInfoBuf    = flatBlendedPerlinInfoBuf;
    }

    public static GpuFusedKernel upload(GpuContext ctx, FusedDensityFunction f) {
        return new GpuFusedKernel(
            f.numKernels,
            buf(ctx, Pointer.to(f.iOpStart),  (long) Sizeof.cl_int    * f.iOpStart.length),
            buf(ctx, Pointer.to(f.dArgStart), (long) Sizeof.cl_int    * f.dArgStart.length),
            buf(ctx, f.flatIOps.length > 0
                    ? Pointer.to(f.flatIOps)
                    : Pointer.to(new int[1]),
                (long) Sizeof.cl_int    * Math.max(1, f.flatIOps.length)),
            buf(ctx, f.flatDArgs.length > 0
                    ? Pointer.to(f.flatDArgs)
                    : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, f.flatDArgs.length)),
            buf(ctx, f.flatNoiseParams.length > 0
                    ? Pointer.to(f.flatNoiseParams)
                    : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, f.flatNoiseParams.length)),
            buf(ctx, f.flatNoiseInfo.length > 0
                    ? Pointer.to(f.flatNoiseInfo)
                    : Pointer.to(new int[1]),
                (long) Sizeof.cl_int    * Math.max(1, f.flatNoiseInfo.length)),
            buf(ctx, f.flatOctaveParams.length > 0
                    ? Pointer.to(f.flatOctaveParams)
                    : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, f.flatOctaveParams.length)),
            buf(ctx, f.flatPermTables.length > 0
                    ? Pointer.to(f.flatPermTables)
                    : Pointer.to(new byte[1]),
                Math.max(1, f.flatPermTables.length)),
            buf(ctx, f.flatSplineHeaders.length > 0
                    ? Pointer.to(f.flatSplineHeaders)
                    : Pointer.to(new int[1]),
                (long) Sizeof.cl_int    * Math.max(1, f.flatSplineHeaders.length)),
            buf(ctx, f.flatSplineFloatPool.length > 0
                    ? Pointer.to(f.flatSplineFloatPool)
                    : Pointer.to(new float[1]),
                (long) Sizeof.cl_float  * Math.max(1, f.flatSplineFloatPool.length)),
            buf(ctx, f.flatSplineChildren.length > 0
                    ? Pointer.to(f.flatSplineChildren)
                    : Pointer.to(new int[1]),
                (long) Sizeof.cl_int    * Math.max(1, f.flatSplineChildren.length)),
            buf(ctx, f.flatBlendedScalars.length > 0
                    ? Pointer.to(f.flatBlendedScalars)
                    : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, f.flatBlendedScalars.length)),
            buf(ctx, f.flatBlendedPerlinFactors.length > 0
                    ? Pointer.to(f.flatBlendedPerlinFactors)
                    : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, f.flatBlendedPerlinFactors.length)),
            buf(ctx, f.flatBlendedPerlinInfo.length > 0
                    ? Pointer.to(f.flatBlendedPerlinInfo)
                    : Pointer.to(new int[1]),
                (long) Sizeof.cl_int    * Math.max(1, f.flatBlendedPerlinInfo.length))
        );
    }

    private static cl_mem buf(GpuContext ctx, Pointer data, long size) {
        return clCreateBuffer(ctx.context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data, null);
    }

    @Override
    public void close() {
        clReleaseMemObject(iOpStartBuf);
        clReleaseMemObject(dArgStartBuf);
        clReleaseMemObject(flatIOpsBuf);
        clReleaseMemObject(flatDArgsBuf);
        clReleaseMemObject(flatNoiseParamsBuf);
        clReleaseMemObject(flatNoiseInfoBuf);
        clReleaseMemObject(flatOctaveParamsBuf);
        clReleaseMemObject(flatPermTablesBuf);
        clReleaseMemObject(flatSplineHeadersBuf);
        clReleaseMemObject(flatSplineFloatPoolBuf);
        clReleaseMemObject(flatSplineChildrenBuf);
        clReleaseMemObject(flatBlendedScalarsBuf);
        clReleaseMemObject(flatBlendedPerlinFactorsBuf);
        clReleaseMemObject(flatBlendedPerlinInfoBuf);
    }
}
