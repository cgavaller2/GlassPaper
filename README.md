# GlassPaper

A fork of [PaperMC](https://github.com/PaperMC/Paper) for Minecraft 1.21.1 that offloads world generation noise computation to the GPU via OpenCL.

---

## What Is This?

Minecraft's terrain generation evaluates a large tree of **density functions** for every cell corner in every chunk. This is embarrassingly parallel — each (x, y, z) position is independent. GlassPaper compiles these density function trees into OpenCL kernels and dispatches them to the GPU, freeing CPU threads for other server work.

---

## Current Status — v0.1

| Feature | Status |
|---|---|
| OpenCL device detection | ✅ Stable |
| ImprovedNoise GPU kernel | ✅ Validated |
| NormalNoise GPU kernel | ✅ Validated |
| Full density function tree GPU VM | ✅ Working |
| Correctness validation on startup | ✅ Passes (max delta ~3.8e-16) |
| Thread-safe dispatch (kernel pool) | ✅ Stable |
| Blended chunk detection | ✅ Correct CPU fallback |
| Cross-chunk batch coalescing | 🔜 Planned (Phase 7) |

---

## How It Works

### Architecture Overview

```
Server startup
  → GpuContext.init()          Enumerate OpenCL platforms, select GPU
  → GpuNoiseKernel.build()     Compile density.cl, create kernel pool (8 per entry point)
  → GpuNoiseValidator          Run 1000-sample correctness check against CPU reference
  → GpuKernelHolder.set()      Make kernel available to chunk gen threads

Chunk generation (per chunk, on Paper worker threads)
  → NoiseChunk.fillSlice()
      → fillSliceGpu()         Collect all cell-corner positions for this slice
          → GpuKernelHolder.getOrUpload()   Compile + upload density function tree (cached)
          → GpuNoiseKernel.evalDensityTreeFast()   GPU dispatch
          → Results written into interpolator slice arrays
          → CPU trilinear interpolation continues unchanged
```

### Density Function Compiler

`DensityFunctionCompiler` walks Minecraft's density function tree and emits a flat bytecode program for a stack-based GPU virtual machine. Supported node types:

- `Noise` — full NormalNoise (two PerlinNoise instances, multi-octave)
- `ShiftedNoise` — noise with XYZ position offsets
- `RangeChoice` — conditional selection based on input range
- `BlendDensity` — identity pass-through (blend values handled at fillSlice level)
- `YClampedGradient` — linear ramp over Y axis
- `Clamp`, `Abs`, `Square`, `Cube`, `HalfNegative`, `QuarterNegative`, `Squeeze`, `Invert`
- `Add`, `Mul`, `Min`, `Max` — binary arithmetic
- `AddConst`, `MulConst` — constant-optimised forms
- `Spline` — CubicSpline evaluation up to 3 levels deep
- `Constant`, `BeardifierMarker` — trivial constants

Unsupported node types fall back to CPU automatically and silently.

### Buffer Strategy

Static data (noise parameters, octave permutation tables, spline data, compiled bytecode) is uploaded to GPU memory **once** per unique density function tree and cached for the lifetime of the server. Only position arrays and output buffers are allocated per dispatch.

### Thread Safety

- `cl_kernel` objects are not thread-safe for concurrent `clSetKernelArg` calls
- GlassPaper maintains a pool of 8 kernel instances per entry point (matching Paper's worker thread count)
- Each thread borrows a kernel from the pool, sets args, dispatches, reads back, and returns the kernel
- The OpenCL command queue is protected by a `synchronized` block covering only the enqueue + blocking read operations

---

## Performance — Current Baseline

Tested on: **NVIDIA GeForce RTX 2080 Ti** + **Intel Core i9-14900K**, Windows 10, Paper 1.21.1

```
GPU dispatches    : ~460,000 per minute of active exploration
GPU avg/dispatch  : ~0.48ms
GPU throughput    : ~490 samples/ms
CPU fallback rate : ~40% (blended chunks at world boundaries — expected)
CPU avg/call      : ~0.10ms
```

### Why GPU and CPU Are Currently at Parity

Each dispatch processes ~235 noise sample points. At this batch size, the fixed overhead of a GPU dispatch (PCIe data transfer, command queue submission, kernel launch latency) consumes approximately 70% of total dispatch time. The GPU is computing correctly and efficiently — it simply doesn't have enough work per dispatch to overcome the launch overhead.

**Batch size needed for GPU to decisively win: ~2000+ points per dispatch.**

### The Path to a Real Speedup

The planned Phase 7 optimization is **cross-chunk batch coalescing** — a `GpuDispatchQueue` that accumulates work from all 8 worker threads simultaneously and flushes one large dispatch when the batch reaches ~4096 points or a timeout elapses. This would eliminate the fixed overhead problem entirely and should yield **3–8× speedup** over the CPU baseline on noise-heavy workloads like initial world generation and Chunky pregeneration.

---

## Requirements

- **Java 21** (Temurin/Adoptium recommended)
- **OpenCL 1.2+** capable GPU or CPU
  - NVIDIA: included with GPU driver (no separate install)
  - AMD: ROCm or AMDGPU-PRO driver
  - Intel: Intel OpenCL Runtime
- **Windows, Linux, or macOS** (JOCL ships native binaries for all platforms)
- Paper 1.21.1 compatible plugins

---

## Building

```bash
git clone https://github.com/YOUR_USERNAME/GlassPaper.git
cd GlassPaper

# Windows
gradlew.bat applyPatches
gradlew.bat createMojmapPaperclipJar

# Linux/macOS
./gradlew applyPatches
./gradlew createMojmapPaperclipJar
```

Output JAR: `paper-server/build/libs/paper-paperclip-1.21.1-R0.1-SNAPSHOT-mojmap.jar`

---

## Running

```bash
java -jar paper-paperclip-1.21.1-R0.1-SNAPSHOT-mojmap.jar --nogui
```

On startup you should see:
```
[GlassPaper] OpenCL device selected: NVIDIA GeForce RTX XXXX
[GlassPaper] OpenCL version: OpenCL 3.0 CUDA
GlassPaper GPU acceleration is active.
[GlassPaper] density.cl compiled and kernel ready.
[GlassPaper] Validation PASSED. 1000 samples, max delta = 3.82e-16
[GlassPaper] NormalNoise validation PASSED. 1000 samples, max delta = 1.03e-15
GlassPaper kernel stored — GPU chunk acceleration enabled.
```

If no OpenCL device is found, GlassPaper falls back to standard Paper CPU generation automatically.

---

## Console Commands

| Command | Description |
|---|---|
| `gpubench` | Print GPU dispatch statistics since last reset |
| `gpureset` | Reset benchmark counters |

---

## Startup Validation

Every time the server starts, GlassPaper runs a 1000-sample correctness check comparing GPU and CPU output for both `ImprovedNoise` and `NormalNoise`. If validation fails, the GPU path is disabled and the server falls back to standard Paper CPU generation. This ensures world corruption from kernel bugs is impossible.

---

## Known Limitations

- **Small batch sizes** — per-chunk dispatches (~235 points) are too small for decisive GPU advantage. Cross-chunk coalescing is planned.
- **Blended chunks** — chunks at the border of existing terrain use CPU generation (blend values are position-dependent and not yet GPU-accelerated).
- **Spline depth limit** — the GPU spline evaluator supports up to 3 levels of nesting. Deeper splines fall back to CPU.
- **`-cl-fast-relaxed-math`** — enabled for performance. Results are within IEEE 754 machine epsilon of CPU output in practice but strict bit-equality is not guaranteed for all inputs.
- **Single command queue** — OpenCL submissions are serialized across threads. A multi-queue architecture is planned alongside batch coalescing.

---

## Roadmap

## Performance — Phase 7 Results

Tested on: **NVIDIA GeForce RTX 2080 Ti** + **Intel Core i9-14900K**, Windows 10
```
GPU dispatches    : ~200,000 per minute of active exploration  
GPU avg/dispatch  : 0.13ms  (was 0.48ms before batch coalescing)
GPU throughput    : 2,884 samples/ms  (was 490 samples/ms — 5.9× improvement)
CPU fallback rate : ~38% (blended chunks at world boundaries — expected)
Batch config      : 256 points threshold, 1ms flush interval
```

### Phase 7 — Cross-Chunk Batch Coalescing (Complete)

Work from all 8 Paper worker threads is accumulated in a shared `GpuDispatchQueue`
and flushed in large batches rather than dispatched per-chunk. This amortizes
the fixed PCIe round-trip cost (~0.35ms) across many chunks simultaneously,
achieving a 5.9× throughput improvement over the Phase 6 baseline.

Batch parameters are tunable at runtime without server restart:
```
gpuconfig <threshold_points> <interval_ms>
gpuconfig 256 1   ← recommended default
gpuconfig 128 0   ← maximum responsiveness (spins one CPU core)
gpuconfig 512 2   ← lower CPU overhead, slightly higher latency
```

### Phase 8 — Multi-Queue Architecture
Per-thread OpenCL command queues alongside the kernel pool to eliminate the remaining queue lock.

### Phase 9 — BlendDensity GPU Support
Port the blending system to GPU to eliminate the ~40% CPU fallback rate on blended chunks.

### Phase 10 — Profiling and Tuning
Use OpenCL profiling events to measure actual kernel execution time vs transfer overhead. Tune local work size and buffer allocation strategy per device.

---

## Credits

Built on [PaperMC](https://github.com/PaperMC/Paper). GPU compute via [JOCL](http://www.jocl.org/). Noise math ported from Minecraft's `net.minecraft.world.level.levelgen.synth` package.
