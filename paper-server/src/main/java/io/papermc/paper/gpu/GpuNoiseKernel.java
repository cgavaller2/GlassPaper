package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.logging.Logger;

public final class GpuNoiseKernel {

    private static final Logger LOGGER = Logger.getLogger("GlassPaper");
    private static final int POOL_SIZE = 8;

    private final GpuContext ctx;
    private final cl_program program;
    private final BlockingQueue<cl_kernel> noisePool        = new LinkedBlockingQueue<>();
    private final BlockingQueue<cl_kernel> normalNoisePool  = new LinkedBlockingQueue<>();
    private final BlockingQueue<cl_kernel> densityTreePool  = new LinkedBlockingQueue<>();

    private GpuNoiseKernel(GpuContext ctx, cl_program program) {
        this.ctx     = ctx;
        this.program = program;
    }

    public static GpuNoiseKernel build() {
        GpuContext ctx = GpuContext.get();
        if (ctx == null) {
            LOGGER.warning("GpuNoiseKernel: no GPU context, skipping.");
            return null;
        }

        String source;
        try (InputStream is = GpuNoiseKernel.class.getResourceAsStream("/gpu/density.cl")) {
            if (is == null) { LOGGER.severe("density.cl not found!"); return null; }
            source = new String(is.readAllBytes(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            LOGGER.severe("Failed to read density.cl: " + e.getMessage());
            return null;
        }

        cl_program program = clCreateProgramWithSource(ctx.context(), 1, new String[]{source}, null, null);
        int buildResult = clBuildProgram(program, 1, new cl_device_id[]{ctx.device()},
            "-cl-mad-enable -cl-fast-relaxed-math", null, null);

        long[] logSize = new long[1];
        clGetProgramBuildInfo(program, ctx.device(), CL_PROGRAM_BUILD_LOG, 0, null, logSize);
        if (logSize[0] > 2) {
            byte[] logBytes = new byte[(int) logSize[0]];
            clGetProgramBuildInfo(program, ctx.device(), CL_PROGRAM_BUILD_LOG,
                logBytes.length, Pointer.to(logBytes), null);
            LOGGER.info("Kernel build log:\n" +
                new String(logBytes, StandardCharsets.UTF_8).trim());
        }

        if (buildResult != CL_SUCCESS) {
            LOGGER.severe("density.cl compilation FAILED (code " + buildResult + ")");
            clReleaseProgram(program);
            return null;
        }

        GpuNoiseKernel result = new GpuNoiseKernel(ctx, program);
        for (int i = 0; i < POOL_SIZE; i++) {
            result.noisePool.add(clCreateKernel(program, "sampleNoise", null));
            result.normalNoisePool.add(clCreateKernel(program, "sampleNormalNoise", null));
            result.densityTreePool.add(clCreateKernel(program, "evalDensityTree", null));
        }

        LOGGER.info("density.cl compiled and kernel ready.");
        return result;
    }

    // ── Borrow / return helpers ───────────────────────────────────────────────

    private cl_kernel borrow(BlockingQueue<cl_kernel> pool) {
        try {
            return pool.take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("GPU kernel pool interrupted", e);
        }
    }

    private void ret(BlockingQueue<cl_kernel> pool, cl_kernel k) {
        pool.add(k);
    }

    // ── sampleBatch (single ImprovedNoise, used for validation) ──────────────

    public double[] sampleBatch(double[] positions, byte[] permTable,
                                double xo, double yo, double zo, int count) {
        cl_kernel k = borrow(noisePool);
        try {
            cl_mem posBuf  = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * positions.length, Pointer.to(positions), null);
            cl_mem permBuf = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                256L, Pointer.to(permTable), null);
            cl_mem outBuf  = clCreateBuffer(ctx.context(), CL_MEM_WRITE_ONLY,
                (long)Sizeof.cl_double * count, null, null);

            clSetKernelArg(k, 0, Sizeof.cl_mem,    Pointer.to(posBuf));
            clSetKernelArg(k, 1, Sizeof.cl_mem,    Pointer.to(permBuf));
            clSetKernelArg(k, 2, Sizeof.cl_double, Pointer.to(new double[]{xo}));
            clSetKernelArg(k, 3, Sizeof.cl_double, Pointer.to(new double[]{yo}));
            clSetKernelArg(k, 4, Sizeof.cl_double, Pointer.to(new double[]{zo}));
            clSetKernelArg(k, 5, Sizeof.cl_mem,    Pointer.to(outBuf));
            clSetKernelArg(k, 6, Sizeof.cl_int,    Pointer.to(new int[]{count}));

            double[] results = new double[count];
            synchronized (ctx.queue()) {
                clEnqueueNDRangeKernel(ctx.queue(), k, 1, null,
                    new long[]{roundUp(count, 64)}, new long[]{64}, 0, null, null);
                clEnqueueReadBuffer(ctx.queue(), outBuf, CL_TRUE, 0,
                    (long)Sizeof.cl_double * count, Pointer.to(results), 0, null, null);
            }

            clReleaseMemObject(posBuf);
            clReleaseMemObject(permBuf);
            clReleaseMemObject(outBuf);
            return results;
        } finally {
            ret(noisePool, k);
        }
    }

    // ── sampleNormalNoiseBatch (used for validation) ──────────────────────────

    public double[] sampleNormalNoiseBatch(double[] positions, NormalNoiseGpuData data, int count) {
        cl_kernel k = borrow(normalNoisePool);
        try {
            cl_mem posBuf    = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * positions.length, Pointer.to(positions), null);
            cl_mem nParamBuf = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * data.noiseParams.length, Pointer.to(data.noiseParams), null);
            cl_mem octBuf    = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * data.octaveParams.length, Pointer.to(data.octaveParams), null);
            cl_mem permBuf   = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                data.permTables.length, Pointer.to(data.permTables), null);
            cl_mem outBuf    = clCreateBuffer(ctx.context(), CL_MEM_WRITE_ONLY,
                (long)Sizeof.cl_double * count, null, null);

            clSetKernelArg(k, 0, Sizeof.cl_mem, Pointer.to(posBuf));
            clSetKernelArg(k, 1, Sizeof.cl_mem, Pointer.to(nParamBuf));
            clSetKernelArg(k, 2, Sizeof.cl_mem, Pointer.to(octBuf));
            clSetKernelArg(k, 3, Sizeof.cl_mem, Pointer.to(permBuf));
            clSetKernelArg(k, 4, Sizeof.cl_mem, Pointer.to(outBuf));
            clSetKernelArg(k, 5, Sizeof.cl_int, Pointer.to(new int[]{data.firstNumOctaves}));
            clSetKernelArg(k, 6, Sizeof.cl_int, Pointer.to(new int[]{data.secondNumOctaves}));
            clSetKernelArg(k, 7, Sizeof.cl_int, Pointer.to(new int[]{count}));

            double[] results = new double[count];
            synchronized (ctx.queue()) {
                clEnqueueNDRangeKernel(ctx.queue(), k, 1, null,
                    new long[]{roundUp(count, 64)}, new long[]{64}, 0, null, null);
                clEnqueueReadBuffer(ctx.queue(), outBuf, CL_TRUE, 0,
                    (long)Sizeof.cl_double * count, Pointer.to(results), 0, null, null);
            }

            clReleaseMemObject(posBuf);
            clReleaseMemObject(nParamBuf);
            clReleaseMemObject(octBuf);
            clReleaseMemObject(permBuf);
            clReleaseMemObject(outBuf);
            return results;
        } finally {
            ret(normalNoisePool, k);
        }
    }

    // ── evalDensityTreeFast (main chunk gen path) ─────────────────────────────

    public double[] evalDensityTreeFast(double[] positions,
                                        GpuCompiledKernel gpuKernel,
                                        int count) {
        cl_kernel k = borrow(densityTreePool);
        try {
            cl_mem posBuf = clCreateBuffer(ctx.context(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long)Sizeof.cl_double * positions.length,
                Pointer.to(positions), null);
            cl_mem outBuf = clCreateBuffer(ctx.context(),
                CL_MEM_WRITE_ONLY,
                (long)Sizeof.cl_double * count,
                null, null);

            clSetKernelArg(k,  0, Sizeof.cl_mem, Pointer.to(posBuf));
            clSetKernelArg(k,  1, Sizeof.cl_mem, Pointer.to(gpuKernel.iOpsBuf));
            clSetKernelArg(k,  2, Sizeof.cl_mem, Pointer.to(gpuKernel.dArgsBuf));
            clSetKernelArg(k,  3, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseParamsBuf));
            clSetKernelArg(k,  4, Sizeof.cl_mem, Pointer.to(gpuKernel.noiseInfoBuf));
            clSetKernelArg(k,  5, Sizeof.cl_mem, Pointer.to(gpuKernel.octaveParamsBuf));
            clSetKernelArg(k,  6, Sizeof.cl_mem, Pointer.to(gpuKernel.permTablesBuf));
            clSetKernelArg(k,  7, Sizeof.cl_mem, Pointer.to(gpuKernel.splineHeadersBuf));
            clSetKernelArg(k,  8, Sizeof.cl_mem, Pointer.to(gpuKernel.splineFloatPoolBuf));
            clSetKernelArg(k,  9, Sizeof.cl_mem, Pointer.to(gpuKernel.splineChildrenBuf));
            clSetKernelArg(k, 10, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedScalarsBuf));
            clSetKernelArg(k, 11, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinFactorsBuf));
            clSetKernelArg(k, 12, Sizeof.cl_mem, Pointer.to(gpuKernel.blendedPerlinInfoBuf));
            clSetKernelArg(k, 13, Sizeof.cl_mem, Pointer.to(outBuf));
            clSetKernelArg(k, 14, Sizeof.cl_int, Pointer.to(new int[]{count}));

            double[] results = new double[count];
            synchronized (ctx.queue()) {
                clEnqueueNDRangeKernel(ctx.queue(), k, 1, null,
                    new long[]{roundUp(count, 64)}, new long[]{64}, 0, null, null);
                clEnqueueReadBuffer(ctx.queue(), outBuf, CL_TRUE, 0,
                    (long)Sizeof.cl_double * count, Pointer.to(results), 0, null, null);
            }

            clReleaseMemObject(posBuf);
            clReleaseMemObject(outBuf);
            return results;
        } finally {
            ret(densityTreePool, k);
        }
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    private static long roundUp(int n, int multiple) {
        return (long)(((n + multiple - 1) / multiple) * multiple);
    }

    public void release() {
        for (cl_kernel k : noisePool)       clReleaseKernel(k);
        for (cl_kernel k : normalNoisePool) clReleaseKernel(k);
        for (cl_kernel k : densityTreePool) clReleaseKernel(k);
        clReleaseProgram(program);
    }
}
