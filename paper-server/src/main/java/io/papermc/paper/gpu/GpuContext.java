package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;
import java.util.logging.Logger;

public final class GpuContext {

    private static GpuContext INSTANCE;
    private static final Logger LOGGER = Logger.getLogger("GlassPaper");

    // Phase 9.5 — per-slot command queues. Combined with multi-flusher
    // dispatch (see GpuDispatchQueue), this lets up to DENSITY_QUEUE_COUNT
    // density-tree kernels execute concurrently on the device.
    //
    // Vendor portability:
    //   - NVIDIA: each queue → CUDA stream; in-order semantics preserved.
    //   - AMD: AMD's OpenCL spawns a driver thread per queue; Southern
    //     Islands+ GPUs round-robin queues across hardware ACEs.
    //     RDNA/CDNA have 4-8 ACEs → 8 queues exploit full HW parallelism.
    //   - Intel: per-thread in-order queues are the standard portable pattern.
    //
    // The original `queue` is retained for the cold-path validators
    // (sampleBatch, sampleNormalNoiseBatch) which run only at startup.
    // Must match GpuNoiseKernel.POOL_SIZE.
    public static final int DENSITY_QUEUE_COUNT = 8;

    private final cl_device_id      device;
    private final cl_context        context;
    private final cl_command_queue  queue;            // validator queue (cold path)
    private final cl_command_queue[] densityQueues;   // hot-path, one per slot

    // Phase 9.6.E — cached at init, consulted on every getOrUpload to refuse
    // density trees whose to-be-__constant buffers exceed the device's limit.
    // OpenCL 1.2 spec minimum is 64 KB; NVIDIA/AMD/Intel all report ≥ 64 KB
    // in practice, but we never assume.
    private final long maxConstantBufferSize;

    private GpuContext(cl_device_id device, cl_context context,
                       cl_command_queue queue, cl_command_queue[] densityQueues,
                       long maxConstantBufferSize) {
        this.device         = device;
        this.context        = context;
        this.queue          = queue;
        this.densityQueues  = densityQueues;
        this.maxConstantBufferSize = maxConstantBufferSize;
    }

    public long maxConstantBufferSize() { return maxConstantBufferSize; }

    public static synchronized GpuContext init() {
        if (INSTANCE != null) return INSTANCE;
        CL.setExceptionsEnabled(true);

        int[] numPlatforms = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);
        if (numPlatforms[0] == 0) {
            LOGGER.warning("No OpenCL platforms found. GPU acceleration disabled.");
            return null;
        }

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        cl_platform_id chosenPlatform = null;
        cl_device_id   chosenDevice   = null;

        outer:
        for (cl_platform_id platform : platforms) {
            for (long deviceType : new long[]{ CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU }) {
                try {
                    int[] count = new int[1];
                    clGetDeviceIDs(platform, deviceType, 0, null, count);
                    if (count[0] == 0) continue;
                    cl_device_id[] devices = new cl_device_id[count[0]];
                    clGetDeviceIDs(platform, deviceType, devices.length, devices, null);
                    chosenPlatform = platform;
                    chosenDevice   = devices[0];
                    break outer;
                } catch (CLException ignored) {}
            }
        }

        if (chosenDevice == null) {
            LOGGER.warning("No usable OpenCL device found. GPU acceleration disabled.");
            return null;
        }

        LOGGER.info("OpenCL device selected: " + queryString(chosenDevice, CL_DEVICE_NAME));
        LOGGER.info("OpenCL version: "         + queryString(chosenDevice, CL_DEVICE_VERSION));

        cl_context_properties props = new cl_context_properties();
        props.addProperty(CL_CONTEXT_PLATFORM, chosenPlatform);
        cl_context ctx = clCreateContext(props, 1, new cl_device_id[]{ chosenDevice }, null, null, null);

        // All queues are in-order. Out-of-order queues (event chains on a
        // single queue) hung clWaitForEvents on NVIDIA in earlier work.
        // Phase 9.5 sidesteps that path: each slot gets its own in-order
        // queue, and multi-flusher submits to them in parallel from
        // separate dispatcher threads — no inter-queue events needed.
        cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, chosenDevice, null, null);

        cl_command_queue[] densityQueues = new cl_command_queue[DENSITY_QUEUE_COUNT];
        for (int i = 0; i < DENSITY_QUEUE_COUNT; i++) {
            densityQueues[i] = clCreateCommandQueueWithProperties(ctx, chosenDevice, null, null);
        }
        LOGGER.info("Created " + DENSITY_QUEUE_COUNT
            + " density command queues for per-slot parallel dispatch.");

        // Vendor-portable capability log. Phase 9.6.E uses __constant for
        // small hot buffers; per-tree size check refuses oversized trees.
        long maxConst = queryUlong(chosenDevice, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
        long maxConstArgs = queryUint(chosenDevice, CL_DEVICE_MAX_CONSTANT_ARGS);
        long maxWg   = queryUlong(chosenDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        LOGGER.info(String.format(
            "Device capabilities: max constant buffer = %.1f KB, "
          + "max constant args = %d, max work-group size = %d",
            maxConst / 1024.0, maxConstArgs, maxWg));

        INSTANCE = new GpuContext(chosenDevice, ctx, queue, densityQueues, maxConst);
        return INSTANCE;
    }

    private static long queryUlong(cl_device_id device, int param) {
        long[] out = new long[1];
        clGetDeviceInfo(device, param, Sizeof.cl_ulong, Pointer.to(out), null);
        return out[0];
    }
    private static long queryUint(cl_device_id device, int param) {
        int[] out = new int[1];
        clGetDeviceInfo(device, param, Sizeof.cl_uint, Pointer.to(out), null);
        return out[0] & 0xFFFFFFFFL;
    }

    public cl_context       context() { return context; }
    public cl_device_id     device()  { return device;  }
    /** Validator queue (cold path). Hot dispatches use a slot's dedicated queue. */
    public cl_command_queue queue()   { return queue;   }
    /** Per-slot density queue at index 0..DENSITY_QUEUE_COUNT-1. */
    public cl_command_queue densityQueue(int index) { return densityQueues[index]; }
    public static GpuContext get()    { return INSTANCE; }

    public void shutdown() {
        for (cl_command_queue q : densityQueues) {
            clReleaseCommandQueue(q);
        }
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        INSTANCE = null;
        LOGGER.info("GPU context released.");
    }

    private static String queryString(cl_device_id device, int param) {
        long[] size = new long[1];
        clGetDeviceInfo(device, param, 0, null, size);
        byte[] buf = new byte[(int) size[0]];
        clGetDeviceInfo(device, param, buf.length, Pointer.to(buf), null);
        return new String(buf, 0, buf.length - 1).trim();
    }
}
