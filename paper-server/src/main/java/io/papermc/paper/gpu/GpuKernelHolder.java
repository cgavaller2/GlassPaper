package io.papermc.paper.gpu;

import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;

public final class GpuKernelHolder {

    private static volatile GpuNoiseKernel instance = null;

    private static volatile GpuDispatchQueue dispatchQueue;

    public static void setDispatchQueue(GpuDispatchQueue queue) { dispatchQueue = queue; }
    public static GpuDispatchQueue getDispatchQueue()           { return dispatchQueue;  }

    // Content-keyed cache: identical compiled trees → one GPU buffer set.
    // Survives across all DensityFunction instances that hash to the same
    // bytecode + static buffers. Required for correctness — see DfCacheKey.
    private static final ConcurrentHashMap<DfCacheKey, GpuCompiledKernel>
        gpuKernelCache = new ConcurrentHashMap<>();

    // Identity-keyed front cache. Skips the (expensive) compile pass when
    // the same DensityFunction instance is queried again — typical for
    // chunk gen, where vanilla MC reuses NoiseInterpolator instances across
    // many chunks. Falls through to the content cache on miss; content cache
    // entries can still de-dup across distinct-but-equivalent fn instances.
    //
    // WeakHashMap so GC can reclaim entries if a density tree is dropped
    // (e.g. world reload). Wrapped in synchronizedMap because WeakHashMap is
    // not thread-safe and getOrUpload runs on every Paper worker thread.
    // Negative entries (compile-failure → null) are stored as a sentinel so
    // we don't recompile a known-unsupported tree on every chunk.
    private static final Object NEGATIVE = new Object();
    private static final Map<net.minecraft.world.level.levelgen.DensityFunction, Object>
        identityCache = java.util.Collections.synchronizedMap(new WeakHashMap<>());

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

        // 0. Identity-cache fast path. Vanilla MC reuses NoiseInterpolator
        // and inner density-function instances across thousands of chunks —
        // skipping the compile pass on hits saves the two-pass tree walk
        // + reflection cost on every interpolator of every slice.
        Object cached = identityCache.get(fn);
        if (cached != null) {
            GlassPaperBenchmark.recordIdentityHit();
            return cached == NEGATIVE ? null : (GpuCompiledKernel) cached;
        }

        // 1. Compile on CPU (cheap if already structurally identical).
        long compileStart = System.nanoTime();
        CompiledDensityFunction cdf =
            net.minecraft.world.level.levelgen.DensityFunctionCompiler.compile(fn);
        if (cdf == null) {
            // Memoize the failure so we don't burn compile cycles on every
            // chunk for unsupported trees.
            identityCache.put(fn, NEGATIVE);
            GlassPaperBenchmark.recordCompileMiss(System.nanoTime() - compileStart);
            return null;
        }

        // 2. Key by content — identical programs share one GPU buffer set.
        DfCacheKey key = DfCacheKey.of(cdf);
        final boolean[] wasHit = { true };
        GpuCompiledKernel result = gpuKernelCache.computeIfAbsent(key, k -> {
            wasHit[0] = false;
            GpuContext ctx = GpuContext.get();
            if (ctx == null) return null;
            return GpuCompiledKernel.upload(ctx, cdf);
        });
        long compileNanos = System.nanoTime() - compileStart;
        if (wasHit[0]) {
            GlassPaperBenchmark.recordCompileHit(compileNanos);
        } else {
            GlassPaperBenchmark.recordCompileMiss(compileNanos);
        }

        if (result != null) {
            identityCache.put(fn, result);
        } else {
            identityCache.put(fn, NEGATIVE);
        }
        return result;
    }

    public static void releaseAll() {
        gpuKernelCache.values().forEach(GpuCompiledKernel::close);
        gpuKernelCache.clear();
        identityCache.clear();
    }
}
