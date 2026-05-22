package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;

public final class GpuCompiledKernel implements AutoCloseable {

    final cl_mem iOpsBuf;
    final cl_mem dArgsBuf;
    final cl_mem noiseParamsBuf;
    final cl_mem noiseInfoBuf;
    final cl_mem octaveParamsBuf;
    final cl_mem permTablesBuf;
    final cl_mem splineHeadersBuf;
    final cl_mem splineFloatPoolBuf;
    final cl_mem splineChildrenBuf;
    final cl_mem blendedScalarsBuf;
    final cl_mem blendedPerlinFactorsBuf;
    final cl_mem blendedPerlinInfoBuf;

    // Phase 9.14.B — retained reference to the source CDF so it can be
    // re-fused with sibling kernels for a fused-dispatch slice fill. The
    // CDF carries the raw bytecode + buffers needed for DensityFunctionFuser
    // to re-index and concatenate. Kept reachable for the lifetime of this
    // GpuCompiledKernel; ~10 KB per unique kernel × ~100 unique kernels per
    // session ≈ ~1 MB extra retention. Negligible.
    public final CompiledDensityFunction cdf;

    private GpuCompiledKernel(
        CompiledDensityFunction cdf,
        cl_mem iOpsBuf, cl_mem dArgsBuf,
        cl_mem noiseParamsBuf, cl_mem noiseInfoBuf,
        cl_mem octaveParamsBuf, cl_mem permTablesBuf,
        cl_mem splineHeadersBuf, cl_mem splineFloatPoolBuf,
        cl_mem splineChildrenBuf,
        cl_mem blendedScalarsBuf, cl_mem blendedPerlinFactorsBuf,
        cl_mem blendedPerlinInfoBuf) {
        this.cdf                    = cdf;
        this.iOpsBuf                = iOpsBuf;
        this.dArgsBuf               = dArgsBuf;
        this.noiseParamsBuf         = noiseParamsBuf;
        this.noiseInfoBuf           = noiseInfoBuf;
        this.octaveParamsBuf        = octaveParamsBuf;
        this.permTablesBuf          = permTablesBuf;
        this.splineHeadersBuf       = splineHeadersBuf;
        this.splineFloatPoolBuf     = splineFloatPoolBuf;
        this.splineChildrenBuf      = splineChildrenBuf;
        this.blendedScalarsBuf      = blendedScalarsBuf;
        this.blendedPerlinFactorsBuf= blendedPerlinFactorsBuf;
        this.blendedPerlinInfoBuf   = blendedPerlinInfoBuf;
    }

    public static GpuCompiledKernel upload(GpuContext ctx, CompiledDensityFunction cdf) {
        return new GpuCompiledKernel(
            cdf,
            buf(ctx, cdf.iOps.length > 0
                    ? Pointer.to(cdf.iOps) : Pointer.to(new int[1]),
                (long) Sizeof.cl_int * Math.max(1, cdf.iOps.length)),
            buf(ctx, cdf.dArgs.length > 0
                    ? Pointer.to(cdf.dArgs) : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, cdf.dArgs.length)),
            buf(ctx, cdf.noiseParams.length > 0
                    ? Pointer.to(cdf.noiseParams) : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, cdf.noiseParams.length)),
            buf(ctx, cdf.noiseInfo.length > 0
                    ? Pointer.to(cdf.noiseInfo) : Pointer.to(new int[1]),
                (long) Sizeof.cl_int * Math.max(1, cdf.noiseInfo.length)),
            buf(ctx, cdf.octaveParams.length > 0
                    ? Pointer.to(cdf.octaveParams) : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, cdf.octaveParams.length)),
            buf(ctx, cdf.permTables.length > 0
                    ? Pointer.to(cdf.permTables) : Pointer.to(new byte[1]),
                Math.max(1, cdf.permTables.length)),
            buf(ctx, cdf.splineHeaders.length > 0
                    ? Pointer.to(cdf.splineHeaders) : Pointer.to(new int[1]),
                (long) Sizeof.cl_int * Math.max(1, cdf.splineHeaders.length)),
            buf(ctx, cdf.splineFloatPool.length > 0
                    ? Pointer.to(cdf.splineFloatPool) : Pointer.to(new float[1]),
                (long) Sizeof.cl_float * Math.max(1, cdf.splineFloatPool.length)),
            buf(ctx, cdf.splineChildren.length > 0
                    ? Pointer.to(cdf.splineChildren) : Pointer.to(new int[1]),
                (long) Sizeof.cl_int * Math.max(1, cdf.splineChildren.length)),
            buf(ctx, cdf.blendedScalars.length > 0
                    ? Pointer.to(cdf.blendedScalars) : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, cdf.blendedScalars.length)),
            buf(ctx, cdf.blendedPerlinFactors.length > 0
                    ? Pointer.to(cdf.blendedPerlinFactors) : Pointer.to(new double[1]),
                (long) Sizeof.cl_double * Math.max(1, cdf.blendedPerlinFactors.length)),
            buf(ctx, cdf.blendedPerlinInfo.length > 0
                    ? Pointer.to(cdf.blendedPerlinInfo) : Pointer.to(new int[1]),
                (long) Sizeof.cl_int * Math.max(1, cdf.blendedPerlinInfo.length))
        );
    }

    private static cl_mem buf(GpuContext ctx, Pointer data, long size) {
        return clCreateBuffer(ctx.context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data, null);
    }

    @Override
    public void close() {
        clReleaseMemObject(iOpsBuf);
        clReleaseMemObject(dArgsBuf);
        clReleaseMemObject(noiseParamsBuf);
        clReleaseMemObject(noiseInfoBuf);
        clReleaseMemObject(octaveParamsBuf);
        clReleaseMemObject(permTablesBuf);
        clReleaseMemObject(splineHeadersBuf);
        clReleaseMemObject(splineFloatPoolBuf);
        clReleaseMemObject(splineChildrenBuf);
        clReleaseMemObject(blendedScalarsBuf);
        clReleaseMemObject(blendedPerlinFactorsBuf);
        clReleaseMemObject(blendedPerlinInfoBuf);
    }
}
