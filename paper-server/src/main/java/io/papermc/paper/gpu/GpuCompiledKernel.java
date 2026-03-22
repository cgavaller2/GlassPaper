package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;

/**
 * Holds pre-uploaded GPU buffers for a CompiledDensityFunction.
 * Created once per unique density function tree, reused for every dispatch.
 * Only positions and output buffers are allocated per-dispatch.
 */
public final class GpuCompiledKernel implements AutoCloseable {

    // Static buffers — uploaded once at construction, never change
    final cl_mem iOpsBuf;
    final cl_mem dArgsBuf;
    final cl_mem noiseParamsBuf;
    final cl_mem noiseInfoBuf;
    final cl_mem octaveParamsBuf;
    final cl_mem permTablesBuf;
    final cl_mem splineHeadersBuf;
    final cl_mem splineFloatPoolBuf;
    final cl_mem splineChildrenBuf;

    private GpuCompiledKernel(
        cl_mem iOpsBuf, cl_mem dArgsBuf,
        cl_mem noiseParamsBuf, cl_mem noiseInfoBuf,
        cl_mem octaveParamsBuf, cl_mem permTablesBuf,
        cl_mem splineHeadersBuf, cl_mem splineFloatPoolBuf,
        cl_mem splineChildrenBuf) {
        this.iOpsBuf           = iOpsBuf;
        this.dArgsBuf          = dArgsBuf;
        this.noiseParamsBuf    = noiseParamsBuf;
        this.noiseInfoBuf      = noiseInfoBuf;
        this.octaveParamsBuf   = octaveParamsBuf;
        this.permTablesBuf     = permTablesBuf;
        this.splineHeadersBuf  = splineHeadersBuf;
        this.splineFloatPoolBuf= splineFloatPoolBuf;
        this.splineChildrenBuf = splineChildrenBuf;
    }

    /** Upload all static buffers to GPU once. */
    public static GpuCompiledKernel upload(GpuContext ctx, CompiledDensityFunction cdf) {
        return new GpuCompiledKernel(
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
                (long) Sizeof.cl_int * Math.max(1, cdf.splineChildren.length))
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
    }
}
