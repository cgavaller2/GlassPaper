# GlassPaper — Agent Context

A fork of PaperMC (Minecraft 1.21.x server) that offloads world generation
density-function evaluation to the GPU via OpenCL. Built with paperweight,
which is non-obvious enough that the build workflow is the most important
thing in this doc.

## Critical: build workflow

Paperweight maintains a **nested git repo** at
`paper-server/src/minecraft/java/.git`. Edits to vanilla Minecraft sources
live there, get committed there, then `rebuildPatches` extracts them as
`.patch` files in the outer repo. **The `src/minecraft/` directory is
gitignored in the outer repo** — there is no other place those changes are
saved.

### The trap that bit a previous session

Running `applyPatches` regenerates `paper-server/src/minecraft/` from
vanilla + the committed `.patch` files. Any edits in `src/minecraft/` that
were not yet captured as patches **are silently destroyed**. Recovery is
possible only through IntelliJ Local History, Windows shadow copies, or
cloud sync — none of which are guaranteed.

### Safe daily workflow

1. Edit anywhere (`src/minecraft/java/...` for vanilla MC overrides,
   `paper-server/src/main/java/io/papermc/paper/gpu/` for GlassPaper's own
   classes, `paper-server/src/main/resources/gpu/density.cl` for the kernel).
2. Build: `gradlew :paper-server:createMojmapPaperclipJar`.
3. Test: `java -jar paper-server/build/libs/paper-paperclip-<ver>-SNAPSHOT-mojmap.jar --nogui`
4. **Before** anything destructive (branch switch, pull, especially
   `applyPatches`): capture edits as patches:
   - `cd paper-server/src/minecraft/java && git add -A && git commit -m "..."`
     (this is the *inner* repo)
   - `gradlew :paper-server:rebuildPatches` (writes/updates `.patch` files
     in `paper-server/patches/` in the outer repo)
   - `git add paper-server/patches && git commit ...` in the outer repo

### When `applyPatches` is needed vs not

- **Needed**: first clone, after pulling upstream that changes patches,
  after hand-editing `.patch` files.
- **Not needed**: between code edits and a build. The build task does not
  touch `src/minecraft/`.

### Build output

`paper-server/build/libs/paper-paperclip-<mc-version>-R0.1-SNAPSHOT-mojmap.jar`
is the runnable paperclip jar. The `paper-bundler-*` and `paper-server-*`
jars in the same dir are intermediates — don't run those.

## Architecture

### Where the GPU plugs into terrain gen

The hook is in `NoiseChunk.fillSlice` (vanilla MC class, modified in
`src/minecraft/java/net/minecraft/world/level/levelgen/NoiseChunk.java`).
Routes to `fillSliceGpu` when `GpuKernelHolder.isAvailable() && blender == Blender.empty()`,
falls back to vanilla CPU loop otherwise. `fillSliceGpu`:

1. Builds a position array for all cell corners in the YZ slice
2. For each interpolator, tries `GpuKernelHolder.getOrUpload(fn)` —
   returns a pre-uploaded `GpuCompiledKernel` if the density tree is
   compilable, else `null` for CPU fallback
3. Submits through `GpuDispatchQueue` (which coalesces work across all 8
   Paper worker threads) and copies results into `interp.slice0/slice1`

The rest of vanilla chunk gen (trilinear interpolation between slice0/slice1)
is unchanged.

### Density function compilation

`net.minecraft.world.level.levelgen.DensityFunctionCompiler` walks the
density function tree and emits flat bytecode (`int[] iOps`, `double[] dArgs`)
plus packed static buffers (noise params, octave permutation tables,
spline data, BlendedNoise data). The bytecode is executed by a stack-based
VM in `density.cl`.

Opcodes: see `io.papermc.paper.gpu.DfOpcode`. Anything unsupported makes
the compiler return `null` → CPU fallback. Compilation failures are
recorded in `GlassPaperBenchmark.recordCompileFailure` for diagnostics.

The compiler uses reflection extensively to extract private fields from
`PerlinNoise`, `NormalNoise`, `BlendedNoise`, and `CubicSpline`. Field
names are hardcoded — they'll break if Mojang renames them (or if the
build's mappings change).

### Buffer caching

`GpuKernelHolder.gpuKernelCache` is keyed by `DfCacheKey` (content hash of
the compiled bytecode), not object identity. Identical density trees share
one GPU buffer set for the server's lifetime. Only the per-dispatch position
and output buffers are allocated each call.

### Kernel pool and dispatch queue

`cl_kernel` is not thread-safe for concurrent `clSetKernelArg`.
`GpuNoiseKernel` keeps a **pool of 8 kernel instances** per entry point
(matching Paper's worker count). Workers borrow a kernel, set args,
dispatch, read back, return it.

`GpuDispatchQueue` (Phase 7) coalesces submissions across threads. The
flusher thread waits on `threshold OR interval` (default 256 points / 1 ms),
groups by `GpuCompiledKernel` identity, merges into one large dispatch
per group, then splits results back to per-worker futures. Tunable at
runtime: `gpuconfig <threshold> <interval_ms>` console command.

### Startup sequence (MinecraftServer)

`MinecraftServer.runServer` (the modified version in `src/minecraft/`)
runs this near the spark-enable point, before delayed init tasks:

1. `GpuContext.init()` — enumerate platforms, pick device, build context
2. `GpuNoiseKernel.build()` — compile `density.cl`, build kernel pool
3. `GpuNoiseValidator.validate()` — 1000-sample ImprovedNoise CPU/GPU diff
4. `GpuNoiseValidator.validateNormalNoise()` — same for NormalNoise
5. `GpuNoiseValidator.validateBlendedNoise()` — 10000-sample BlendedNoise
   diff with a synthetic test tree that deliberately places the real
   BlendedNoise at high octave offsets (the test exists because there was
   an octave-offset bug at some point)
6. If validation passes: `GpuKernelHolder.set(kernel)` and start the dispatch queue
7. On any validation failure: log warn, leave `GpuKernelHolder` null → CPU only

### Console commands (added in DedicatedServer)

- `gpubench` — print `GlassPaperBenchmark.report()`
- `gpureset` — zero counters
- `gpuconfig <threshold> <interval_ms>` — retune dispatch queue at runtime
- `gpuvalidate` — flip a flag for runtime CPU/GPU comparison (the
  comparison code path was half-installed; check before relying on it)

### Shutdown (MinecraftServer)

Prints `GlassPaperBenchmark.report()`, shuts the dispatch queue, releases
cached GPU buffers, shuts down `GpuContext`.

## Bug history and current state

### Fixed in Phase 9.1 (commit "GlassPaper: Phase 9.1 cache-key fix...")

**Cache-key collision (root cause of the curved-volume terrain artifacts).**
`DfCacheKey` originally hashed only `iOps + dArgs`. Each compile assigns
local NormalNoise indices `0, 1, 2, ...`, so structurally-identical density
trees (e.g. multiple cave functions with one `Noise` node each) produced
bytecode-identical `iOps`. The cache returned the first compile's
`GpuCompiledKernel` — uploaded with the *first* noise's permutation tables —
for every later collision. GPU sampled the wrong noise, terrain looked
"noise-like but wrong". Fixed by extending the cache key to include all
static buffer contents (`noiseParams`, `noiseInfo`, `octaveParams`,
`permTables`, spline arrays, blended-noise arrays). See
`DfCacheKey.java` for the rationale comment.

**Zero-filled futures on dispatch exception.** `GpuDispatchQueue.flush()`
exception path used to call `future.complete(new double[count])`, and the
caller in `NoiseChunk.fillSliceGpu` treats any non-null result as a valid
GPU dispatch. Zero-filled slice would `arraycopy` into the interpolator
and produce flat/missing terrain. Now completes with `null`, matching the
timeout path the caller already handles correctly. Same fix applied to
the `shutdown()` drain path.

**WeirdScaledSampler silent enum fallback.** Was
`ws.rarityValueMapper().name().equals("TYPE1") ? 0 : 1`, which silently
used TYPE2 for any unknown enum name. Now fails closed → CPU fallback,
catching obfuscated-build and version-update breakage.

**Compile-failure log spam.** Each per-chunk compile failure used to log
a WARN. Now logs once per distinct reason; counts the rest. Counters
appear in `gpubench` under "Compile failures by reason".

**`[GlassPaper] [GlassPaper]` double prefix.** The logger name and the
message both prepended `[GlassPaper]`. Stripped from the message strings.

### Validation infrastructure (Phase 9.1)

`gpuvalidate` console command toggles a `GlassPaperBenchmark.validating`
flag. When on, `NoiseChunk.fillSliceGpu` samples every 16th point per
dispatch, evaluates it on CPU via `SinglePointContext`, and records any
delta > 1e-6 against the density function's class name. `gpubench`
prints per-type stats: total mismatches, sample count, max delta with
position, average delta. For RangeChoice mismatches specifically, the
first 10 are rich-diagnosed via reflection — extracts `input`, `min`,
`max`, `whenInRange`, `whenOutOfRange`, evaluates each on CPU, and logs
which branch the GPU output most likely matches. This is what found the
cache-key bug. Keep it in mind for any future divergence investigation.

### Currently disabled / open

**BlendedNoise GPU compilation** is hardcoded-disabled at
`DensityFunctionCompiler.java` around line 400:
```java
if (fn instanceof BlendedNoise bn) {
    fail("BlendedNoise: temporarily disabled for debugging");
    return;
}
```
This was disabled during a previous debugging session and never re-enabled.
Result: every density tree containing BlendedNoise falls back to CPU (~50%
of dispatches). Re-enabling is the obvious next step — the validation
infrastructure can now catch BlendedNoise correctness issues automatically.
The `GpuNoiseValidator.validateBlendedNoise` startup check exists but is
not called from `MinecraftServer` in the current recovered code (Local
History gave back a version without the call).

**CacheOnce / CacheAllInCell bypass.** `fillSliceGpu` writes directly into
`interp.slice0/slice1` via `arraycopy` instead of calling
`interp.fillArray(...)`. The interpolator's wrapped tree may contain
`CacheOnce` or `CacheAllInCell` nodes that vanilla relies on for
downstream cache hits in `selectCellYZ → cacheAllInCell.noiseFiller.fillArray`.
Verified non-correctness-impacting (`arrayInterpolationCounter` can't
collide with stale values), but leaves CPU work on the table. Possible
follow-up perf win.

## File map

GlassPaper-specific code:
- `paper-server/src/main/java/io/papermc/paper/gpu/` — all GPU classes
  (GpuContext, GpuNoiseKernel, GpuCompiledKernel, GpuKernelHolder,
  GpuDispatchQueue, GpuWorkItem, CompiledDensityFunction, DfCacheKey,
  DfOpcode, GpuNoiseValidator, NormalNoiseGpuData, GlassPaperBenchmark)
- `paper-server/src/main/resources/gpu/density.cl` — the OpenCL kernel

Vanilla MC files modified (in inner repo at `paper-server/src/minecraft/java/`):
- `net/minecraft/server/MinecraftServer.java` — startup/shutdown hooks
- `net/minecraft/server/dedicated/DedicatedServer.java` — console commands
- `net/minecraft/world/level/levelgen/NoiseChunk.java` — `fillSliceGpu`
- `net/minecraft/world/level/levelgen/DensityFunctionCompiler.java` — new file

After `rebuildPatches`, all the above modifications + the new file are
bundled into one feature patch:
`paper-server/patches/features/00NN-GlassPaper-GPU-accelerated-terrain-generation-integr.patch`

## Performance snapshot

RTX 2080 Ti + i9-14900K, Windows.

Phase 9.1 (current, cache-key fix in, BlendedNoise still CPU-only):
- ~2,112 samples/ms throughput
- ~0.17 ms avg/dispatch
- ~51% CPU fallback rate — driven entirely by BlendedNoise being disabled.
  Re-enabling BlendedNoise on GPU should reclaim most of this.

Phase 7 (Cross-chunk batch coalescing, before the cache-key bug was found
or fixed — these numbers were achievable but with terrain artifacts):
- ~2,884 samples/ms throughput
- ~0.13 ms avg/dispatch
- ~38% CPU fallback rate

The intent is to get back to (and past) Phase 7 throughput now that the
correctness bugs are out of the way — re-enable BlendedNoise, then look
at the CacheOnce bypass and the remaining queue lock.

## Requirements

- Java 21
- OpenCL 1.2+ device (NVIDIA driver includes runtime; AMD needs ROCm or
  AMDGPU-PRO; Intel needs Intel OpenCL Runtime)
- JOCL 2.0.5 (declared in `paper-server/build.gradle.kts`)
- Windows / Linux / macOS (JOCL ships native binaries for all three)

## When in doubt

- "How does X get from CPU density tree to GPU?" — `DensityFunctionCompiler.compile`
  emits bytecode, `GpuCompiledKernel.upload` uploads it, `density.cl`'s
  `evalDensityTree` kernel executes it.
- "Where does GPU output flow back into the world?" — `NoiseChunk.fillSliceGpu`
  → `System.arraycopy` into `interp.slice0/slice1` → vanilla trilinear
  interpolation.
- "What density function nodes are supported?" — the `instanceof` chain
  in `DensityFunctionCompiler.emit`. Unsupported → `fail(...)` → CPU fallback.
- "Why is the GPU path not engaging?" — check
  `GpuKernelHolder.isAvailable()` (startup validation), check the blender
  is empty (`blender == Blender.empty()`), check the density tree compiles
  (`GlassPaperBenchmark.report()` shows compile-failure counts).
