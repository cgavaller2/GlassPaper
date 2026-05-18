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
    public static boolean isAvailable()           { return instance != null; }

    private static volatile boolean blendDensityLogged = false;

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

        if (fn.getClass().getSimpleName().equals("BlendDensity") && !blendDensityLogged) {
            blendDensityLogged = true;
            StringBuilder sb = new StringBuilder("[GlassPaper] BlendDensity iOps: ");
            for (int op : cdf.iOps) sb.append(op).append(",");
            java.util.logging.Logger.getLogger("GlassPaper").info(sb.toString());
            sb = new StringBuilder("[GlassPaper] BlendDensity dArgs count: " + cdf.dArgs.length
                + " octaves: " + (cdf.octaveParams.length/4));
            java.util.logging.Logger.getLogger("GlassPaper").info(sb.toString());
        }

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
