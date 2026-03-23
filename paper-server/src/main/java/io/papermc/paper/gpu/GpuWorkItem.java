package io.papermc.paper.gpu;

import java.util.concurrent.CompletableFuture;

/**
 * A single unit of GPU work submitted by a worker thread.
 * The submitting thread blocks on future.get() until the flusher completes it.
 */
public final class GpuWorkItem {

    /** Flat position array: [x0,y0,z0, x1,y1,z1, ...] length = count*3 */
    public final double[] positions;

    /** Pre-uploaded GPU buffers for this density function tree */
    public final GpuCompiledKernel gpuKernel;

    /** Number of sample points */
    public final int count;

    /** Offset into the merged batch position array assigned by the flusher */
    public volatile int batchOffset;

    /** Completed by the flusher thread with this item's slice of the batch result */
    public final CompletableFuture<double[]> future = new CompletableFuture<>();

    public GpuWorkItem(double[] positions, GpuCompiledKernel gpuKernel, int count) {
        this.positions = positions;
        this.gpuKernel = gpuKernel;
        this.count     = count;
    }
}
