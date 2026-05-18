package io.papermc.paper.gpu;

import org.jocl.*;
import static org.jocl.CL.*;
import java.util.logging.Logger;

public final class GpuContext {

    private static GpuContext INSTANCE;
    private static final Logger LOGGER = Logger.getLogger("GlassPaper");

    private final cl_device_id device;
    private final cl_context context;
    private final cl_command_queue queue;

    private GpuContext(cl_device_id device, cl_context context, cl_command_queue queue) {
        this.device  = device;
        this.context = context;
        this.queue   = queue;
    }

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
        cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, chosenDevice, null, null);

        INSTANCE = new GpuContext(chosenDevice, ctx, queue);
        return INSTANCE;
    }

    public cl_context       context() { return context; }
    public cl_device_id     device()  { return device;  }
    public cl_command_queue queue()   { return queue;   }
    public static GpuContext get()    { return INSTANCE; }

    public void shutdown() {
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
