package io.papermc.paper.gpu;

import java.util.concurrent.ConcurrentHashMap;

public final class GpuKernelHolder {

    private static volatile GpuNoiseKernel instance = null;

    private static volatile GpuDispatchQueue dispatchQueue;

    public static void setDispatchQueue(GpuDispatchQueue queue) { dispatchQueue = queue; }
    public static GpuDispatchQueue getDispatchQueue()           { return dispatchQueue;  }

    // Content-keyed cache: identical compiled trees → one GPU buffer set
    private static final ConcurrentHashMap<DfCacheKey, GpuCompiledKernel>
        gpuKernelCache = new ConcurrentHashMap<>();

    private GpuKernelHolder() {}

    public static void set(GpuNoiseKernel kernel) { instance = kernel; }
    public static GpuNoiseKernel get()            { return instance; }

    // Runtime kill switch — when true, fillSliceGpu acts as if GPU is
    // unavailable and runs everything on CPU. Useful for A/B perf measurement.
    // Initialized from -Dglasspaper.gpu.disable=true at JVM startup so
    // pregeneration runs can compare GPU vs CPU without rebuilds.
    private static volatile boolean disabled =
        Boolean.getBoolean("glasspaper.gpu.disable");

    public static boolean isDisabled() { return disabled; }
    public static void    setDisabled(boolean d) { disabled = d; }

    public static boolean isAvailable() {
        return instance != null && !disabled;
    }

    /**
     * Compile the density function and return pre-uploaded GPU buffers.
     * Identical trees reuse the same GPU buffers indefinitely.
     * Returns null if the function is unsupported.
     */
    public static GpuCompiledKernel getOrUpload(
        net.minecraft.world.level.levelgen.DensityFunction fn) {
        if (instance == null) return null;

        // 1. Compile on CPU (cheap if already structurally identical)
        CompiledDensityFunction cdf =
            net.minecraft.world.level.levelgen.DensityFunctionCompiler.compile(fn);
        if (cdf == null) return null;

        // 2. Key by content — identical programs share one GPU buffer set
        DfCacheKey key = DfCacheKey.of(cdf);

        return gpuKernelCache.computeIfAbsent(key, k -> {
            GpuContext ctx = GpuContext.get();
            if (ctx == null) return null;
            return GpuCompiledKernel.upload(ctx, cdf);
        });
    }

    public static void releaseAll() {
        gpuKernelCache.values().forEach(GpuCompiledKernel::close);
        gpuKernelCache.clear();
    }
}
