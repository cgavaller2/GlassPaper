/*
 * density.cl  —  GlassPaper GPU density function evaluator
 *
 * Implements:
 *  1. ImprovedNoise / PerlinNoise / NormalNoise primitives
 *  2. A stack-based VM that evaluates compiled density function trees
 *  3. A 3-level spline evaluator (Minecraft CubicSpline)
 */

// ═══════════════════════════════════════════════════════════════════
//  SECTION 1 — Noise primitives (unchanged from before)
// ═══════════════════════════════════════════════════════════════════

__constant int GRADIENT[16][3] = {
    { 1,  1,  0},{-1,  1,  0},{ 1, -1,  0},{-1, -1,  0},
    { 1,  0,  1},{-1,  0,  1},{ 1,  0, -1},{-1,  0, -1},
    { 0,  1,  1},{ 0, -1,  1},{ 0,  1, -1},{ 0, -1, -1},
    { 1,  1,  0},{ 0, -1,  1},{-1,  1,  0},{ 0, -1, -1}
};

static double smoothstep(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
static double lerp(double t, double a, double b)  { return a + t * (b - a); }
static double lerp2(double tx, double ty, double x0, double x1, double y0, double y1) {
    return lerp(ty, lerp(tx, x0, x1), lerp(tx, y0, y1));
}
static double lerp3(double tx, double ty, double tz,
                    double v000, double v100, double v010, double v110,
                    double v001, double v101, double v011, double v111) {
    return lerp(tz, lerp2(tx,ty,v000,v100,v010,v110),
                    lerp2(tx,ty,v001,v101,v011,v111));
}
static double wrapNoise(double v) {
    return v - floor(v / 33554432.0 + 0.5) * 33554432.0;
}
static int perm(int idx, __global const uchar* p) { return (int)(p[idx & 0xFF]); }
static double gradDot(int gi, double x, double y, double z) {
    gi &= 15;
    return (double)GRADIENT[gi][0]*x + (double)GRADIENT[gi][1]*y + (double)GRADIENT[gi][2]*z;
}
static double sampleAndLerp(int gx, int gy, int gz,
                             double dx, double wdy, double dz, double dy,
                             __global const uchar* p) {
    int i  = perm(gx,   p), i1 = perm(gx+1, p);
    int i2 = perm(i +gy,p), i3 = perm(i +gy+1,p);
    int i4 = perm(i1+gy,p), i5 = perm(i1+gy+1,p);
    double d  = gradDot(perm(i2+gz,  p), dx,     wdy,     dz    );
    double d1 = gradDot(perm(i4+gz,  p), dx-1.0, wdy,     dz    );
    double d2 = gradDot(perm(i3+gz,  p), dx,     wdy-1.0, dz    );
    double d3 = gradDot(perm(i5+gz,  p), dx-1.0, wdy-1.0, dz    );
    double d4 = gradDot(perm(i2+gz+1,p), dx,     wdy,     dz-1.0);
    double d5 = gradDot(perm(i4+gz+1,p), dx-1.0, wdy,     dz-1.0);
    double d6 = gradDot(perm(i3+gz+1,p), dx,     wdy-1.0, dz-1.0);
    double d7 = gradDot(perm(i5+gz+1,p), dx-1.0, wdy-1.0, dz-1.0);
    return lerp3(smoothstep(dx), smoothstep(dy), smoothstep(dz),
                 d,d1,d2,d3,d4,d5,d6,d7);
}
static double improvedNoise(double x, double y, double z,
                             double xo, double yo, double zo,
                             __global const uchar* p) {
    double dx=x+xo, dy=y+yo, dz=z+zo;
    int gx=(int)floor(dx), gy=(int)floor(dy), gz=(int)floor(dz);
    double fx=dx-gx, fy=dy-gy, fz=dz-gz;
    return sampleAndLerp(gx,gy,gz, fx,fy,fz,fy, p);
}
static double perlinGetValue(double x, double y, double z,
                              double inputF, double valueF, int nOct, int octBase,
                              __global const double* octP,
                              __global const uchar*  perms) {
    double result=0.0, is=inputF, vs=valueF;
    for (int i=0; i<nOct; i++) {
        int pb = (octBase+i)*4;
        double amp = octP[pb];
        if (amp != 0.0) {
            __global const uchar* pt = perms + (long)(octBase+i)*256;
            result += amp * improvedNoise(wrapNoise(x*is), wrapNoise(y*is), wrapNoise(z*is),
                                          octP[pb+1], octP[pb+2], octP[pb+3], pt) * vs;
        }
        is *= 2.0; vs /= 2.0;
    }
    return result;
}
static double evalNoise(int ni, double x, double y, double z,
                         __global const double* noiseParams,
                         __global const int*    noiseInfo,
                         __global const double* octaveParams,
                         __global const uchar*  permTables) {
    int pb=ni*5, ib=ni*4;
    double vf=noiseParams[pb], fiF=noiseParams[pb+1], fvF=noiseParams[pb+2],
           siF=noiseParams[pb+3], svF=noiseParams[pb+4];
    int foOff=noiseInfo[ib], soOff=noiseInfo[ib+1], fc=noiseInfo[ib+2], sc=noiseInfo[ib+3];
    double fv = perlinGetValue(x,y,z,fiF,fvF,fc,foOff,octaveParams,permTables);
    double sv = perlinGetValue(x*1.0181268882175227, y*1.0181268882175227, z*1.0181268882175227,
                               siF,svF,sc,soOff,octaveParams,permTables);
    return (fv+sv)*vf;
}

// ── ImprovedNoise.noise(x,y,z,yScale,yMax) — full deprecated path ────────────
// Used by BlendedNoise internally
static double improvedNoiseYScale(
    double x, double y, double z,
    double xo, double yo, double zo,
    double yScale, double yMax,
    __global const uchar* p)
{
    double dx = x + xo;
    double dy = y + yo;
    double dz = z + zo;
    int gx = (int)floor(dx);
    int gy = (int)floor(dy);
    int gz = (int)floor(dz);
    double deltaX = dx - gx;
    double deltaY = dy - gy;
    double deltaZ = dz - gz;

    double d7;
    if (yScale != 0.0) {
        double d6 = (yMax >= 0.0 && yMax < deltaY) ? yMax : deltaY;
        d7 = floor(d6 / yScale + 1.0e-7f) * yScale;
    } else {
        d7 = 0.0;
    }
    return sampleAndLerp(gx, gy, gz, deltaX, deltaY - d7, deltaZ, deltaY, p);
}

// ── PerlinNoise.getValue with yScale/yMax (used by BlendedNoise) ─────────────
static double perlinGetValueYScale(
    double x, double y, double z,
    double yScale, double yMax,
    double inputF, double valueF,
    int nOct, int octBase,
    __global const double* octP,
    __global const uchar*  perms)
{
    double result = 0.0;
    double is = inputF, vs = valueF;
    for (int i = 0; i < nOct; i++) {
        int pb = (octBase + i) * 4;
        double amp = octP[pb];
        if (amp != 0.0) {
            __global const uchar* pt = perms + (long)(octBase + i) * 256;
            double xo = octP[pb+1], yo = octP[pb+2], zo = octP[pb+3];
            double n = improvedNoiseYScale(
                wrapNoise(x * is), wrapNoise(y * is), wrapNoise(z * is),
                xo, yo, zo,
                yScale * is, yMax * is, pt);
            result += amp * n * vs;
        }
        is *= 2.0; vs /= 2.0;
    }
    return result;
}

// ── BlendedNoise.compute(x,y,z) ──────────────────────────────────────────────
// bi = blended noise index
// blendedScalars:       5 doubles per instance [xzMul, yMul, xzFac, yFac, smear]
// blendedPerlinFactors: 6 doubles per instance [minIF, minVF, maxIF, maxVF, mainIF, mainVF]
// blendedPerlinInfo:    6 ints per instance    [minOff, maxOff, mainOff, minCnt, maxCnt, mainCnt]
static double evalBlendedNoise(
    int bi, double x, double y, double z,
    __global const double* blendedScalars,
    __global const double* blendedPerlinFactors,
    __global const int*    blendedPerlinInfo,
    __global const double* octaveParams,
    __global const uchar*  permTables)
{
    int sb = bi * 5;
    double xzMul  = blendedScalars[sb+0];
    double yMul   = blendedScalars[sb+1];
    double xzFac  = blendedScalars[sb+2];
    double yFac   = blendedScalars[sb+3];
    double smear  = blendedScalars[sb+4];

    int fb = bi * 6;
    double minIF  = blendedPerlinFactors[fb+0];
    double minVF  = blendedPerlinFactors[fb+1];
    double maxIF  = blendedPerlinFactors[fb+2];
    double maxVF  = blendedPerlinFactors[fb+3];
    double mainIF = blendedPerlinFactors[fb+4];
    double mainVF = blendedPerlinFactors[fb+5];

    int ib = bi * 6;
    int minOff   = blendedPerlinInfo[ib+0];
    int maxOff   = blendedPerlinInfo[ib+1];
    int mainOff  = blendedPerlinInfo[ib+2];
    int minCnt   = blendedPerlinInfo[ib+3];
    int maxCnt   = blendedPerlinInfo[ib+4];
    int mainCnt  = blendedPerlinInfo[ib+5];

    double d  = x * xzMul;
    double d1 = y * yMul;
    double d2 = z * xzMul;
    double d3 = d  / xzFac;
    double d4 = d1 / yFac;
    double d5 = d2 / xzFac;
    double d6 = yMul * smear;
    double d7 = d6  / yFac;

    // ── Main noise (8 octaves) ────────────────────────────────────────────────
    double d10 = 0.0;
    double scale = 1.0;
    for (int i = 0; i < 8 && i < mainCnt; i++) {
            int pb = (mainOff + (mainCnt - 1 - i)) * 4;
            double amp = octaveParams[pb];
            if (amp != 0.0) {
                __global const uchar* pt = permTables + (long)(mainOff + (mainCnt - 1 - i)) * 256;
            double xo = octaveParams[pb+1], yo = octaveParams[pb+2], zo = octaveParams[pb+3];
            d10 += improvedNoiseYScale(
                wrapNoise(d3 * scale), wrapNoise(d4 * scale), wrapNoise(d5 * scale),
                xo, yo, zo,
                d7 * scale, d4 * scale, pt) / scale;
        }
        scale /= 2.0;
    }

    double d12 = (d10 / 10.0 + 1.0) / 2.0;
    d12 = clamp(d12, 0.0, 1.0);
    int flag1 = (d12 >= 1.0) ? 1 : 0;
    int flag2 = (d12 <= 0.0) ? 1 : 0;

    // ── Min/Max limit noise (16 octaves each) ─────────────────────────────────
    double d8 = 0.0, d9 = 0.0;
    scale = 1.0;
    for (int i1 = 0; i1 < 16 && (i1 < minCnt || i1 < maxCnt); i1++) {
        double d13 = wrapNoise(d  * scale);
        double d14 = wrapNoise(d1 * scale);
        double d15 = wrapNoise(d2 * scale);
        double d16 = d6 * scale;

        if (!flag1 && i1 < minCnt) {
            int pb = (minOff + (minCnt - 1 - i1)) * 4;
            double amp = octaveParams[pb];
            if (amp != 0.0) {
                __global const uchar* pt = permTables + (long)(minOff + (minCnt - 1 - i1)) * 256;
                double xo = octaveParams[pb+1], yo = octaveParams[pb+2], zo = octaveParams[pb+3];
                d8 += improvedNoiseYScale(d13, d14, d15, xo, yo, zo,
                    d16, d1 * scale, pt) / scale;
            }
        }
        if (!flag2 && i1 < maxCnt) {
            int pb = (maxOff + (maxCnt - 1 - i1)) * 4;
            double amp = octaveParams[pb];
            if (amp != 0.0) {
                __global const uchar* pt = permTables + (long)(maxOff + (maxCnt - 1 - i1)) * 256;
                double xo = octaveParams[pb+1], yo = octaveParams[pb+2], zo = octaveParams[pb+3];
                d9 += improvedNoiseYScale(d13, d14, d15, xo, yo, zo,
                    d16, d1 * scale, pt) / scale;
            }
        }
        scale /= 2.0;
    }

    // clampedLerp(d12, d8/512, d9/512) / 128
    double lo = d8 / 512.0;
    double hi = d9 / 512.0;
    double result;
    if (d12 <= 0.0) result = lo;
    else if (d12 >= 1.0) result = hi;
    else result = lo + d12 * (hi - lo);
    return result / 128.0;
}

// ── NoiseRouterData.QuantizedSpaghettiRarity ──────────────────────────────────
static double getSpaghettiRarity3D(double v) {
    if (v < -0.5) return 0.75;
    if (v <  0.0) return 1.0;
    if (v <  0.5) return 1.5;
    return 2.0;
}

static double getSphaghettiRarity2D(double v) {
    if (v < -0.75) return 0.5;
    if (v < -0.5)  return 0.75;
    if (v <  0.5)  return 1.0;
    if (v <  0.75) return 2.0;
    return 3.0;
}

// ═══════════════════════════════════════════════════════════════════
//  SECTION 2 — Spline evaluator (depth-limited, 3 levels)
// ═══════════════════════════════════════════════════════════════════

// Spline header layout (4 ints per spline):
//   [0] type: 0=constant, 1=multipoint+float values, 2=multipoint+spline children
//   [1] knotCount
//   [2] fpStart: floatPool index for locations[K] + derivatives[K]
//   [3] extra:   type1→valStart in floatPool, type2→csStart in childIndices, type0→unused

static float hermiteInterp(float coord, int lo, int K, int fpStart, float vLo, float vHi,
                             __global const float* fp) {
    float locLo = fp[fpStart+lo], locHi = fp[fpStart+lo+1];
    float dLo   = fp[fpStart+K+lo], dHi  = fp[fpStart+K+lo+1];
    float span  = locHi - locLo;
    float t     = (coord - locLo) / span;
    float o     = dLo*span - (vHi-vLo);
    float p     = -dHi*span + (vHi-vLo);
    return mix(vLo, vHi, t) + t*(1.0f-t)*mix(o, p, t);
}

static int findKnot(float coord, int K, int fpStart, __global const float* fp) {
    int lo = K;
    for (int i = 0; i < K; i++) { if (coord < fp[fpStart+i]) { lo=i; break; } }
    return lo - 1;
}

// Forward declarations for depth-limited recursion
static float evalSplineD0(int idx, float c0,
    __global const int* sh, __global const float* fp, __global const int* ci);
static float evalSplineD1(int idx, float c0, float c1,
    __global const int* sh, __global const float* fp, __global const int* ci);
static float evalSplineD2(int idx, float c0, float c1, float c2,
    __global const int* sh, __global const float* fp, __global const int* ci);
static float evalSplineD3(int idx, float c0, float c1, float c2, float c3,
    __global const int* sh, __global const float* fp, __global const int* ci);

static float evalSplineD0(int idx, float c0,
    __global const int* sh, __global const float* fp, __global const int* ci)
{
    int b = idx*4, type=sh[b], K=sh[b+1], fpS=sh[b+2];
    if (type == 0) return fp[fpS];  // constant
    if (type == 1) {
        int vS = sh[b+3];
        int lo = findKnot(c0, K, fpS, fp);
        if (lo < 0)   return fp[vS]       + fp[fpS+K]       * (c0 - fp[fpS]);
        if (lo >= K-1) return fp[vS+K-1]  + fp[fpS+2*K-1]  * (c0 - fp[fpS+K-1]);
        return hermiteInterp(c0, lo, K, fpS, fp[vS+lo], fp[vS+lo+1], fp);
    }
    return 0.0f; // type2 at depth0 shouldn't happen
}

static float evalSplineD1(int idx, float c0, float c1,
    __global const int* sh, __global const float* fp, __global const int* ci)
{
    int b=idx*4, type=sh[b], K=sh[b+1], fpS=sh[b+2];
    if (type == 0) return fp[fpS];
    if (type == 1) {
        int vS = sh[b+3];
        int lo = findKnot(c0, K, fpS, fp);
        if (lo < 0)    return fp[vS]      + fp[fpS+K]      * (c0 - fp[fpS]);
        if (lo >= K-1) return fp[vS+K-1]  + fp[fpS+2*K-1] * (c0 - fp[fpS+K-1]);
        return hermiteInterp(c0, lo, K, fpS, fp[vS+lo], fp[vS+lo+1], fp);
    }
    // type 2: children evaluated with d0 using c1
    int csS = sh[b+3];
    int lo  = findKnot(c0, K, fpS, fp);
    float vLo, vHi;
    if (lo < 0) {
        vLo = evalSplineD0(ci[csS], c1, sh, fp, ci);
        return vLo + fp[fpS+K] * (c0 - fp[fpS]);
    }
    if (lo >= K-1) {
        vLo = evalSplineD0(ci[csS+K-1], c1, sh, fp, ci);
        return vLo + fp[fpS+2*K-1] * (c0 - fp[fpS+K-1]);
    }
    vLo = evalSplineD0(ci[csS+lo],   c1, sh, fp, ci);
    vHi = evalSplineD0(ci[csS+lo+1], c1, sh, fp, ci);
    return hermiteInterp(c0, lo, K, fpS, vLo, vHi, fp);
}

static float evalSplineD2(int idx, float c0, float c1, float c2,
    __global const int* sh, __global const float* fp, __global const int* ci)
{
    int b=idx*4, type=sh[b], K=sh[b+1], fpS=sh[b+2];
    if (type == 0) return fp[fpS];
    if (type == 1) {
        int vS = sh[b+3];
        int lo = findKnot(c0, K, fpS, fp);
        if (lo < 0)    return fp[vS]      + fp[fpS+K]      * (c0 - fp[fpS]);
        if (lo >= K-1) return fp[vS+K-1]  + fp[fpS+2*K-1] * (c0 - fp[fpS+K-1]);
        return hermiteInterp(c0, lo, K, fpS, fp[vS+lo], fp[vS+lo+1], fp);
    }
    int csS = sh[b+3];
    int lo  = findKnot(c0, K, fpS, fp);
    float vLo, vHi;
    if (lo < 0) {
        vLo = evalSplineD1(ci[csS], c1, c2, sh, fp, ci);
        return vLo + fp[fpS+K] * (c0 - fp[fpS]);
    }
    if (lo >= K-1) {
        vLo = evalSplineD1(ci[csS+K-1], c1, c2, sh, fp, ci);
        return vLo + fp[fpS+2*K-1] * (c0 - fp[fpS+K-1]);
    }
    vLo = evalSplineD1(ci[csS+lo],   c1, c2, sh, fp, ci);
    vHi = evalSplineD1(ci[csS+lo+1], c1, c2, sh, fp, ci);
    return hermiteInterp(c0, lo, K, fpS, vLo, vHi, fp);
}

// Vanilla MC overworld splines nest 4 deep (continentalness -> erosion ->
// weirdness -> ridges). The earlier 3-level cap silently truncated the
// deepest coordinate, producing wrong terrain shape that propagated through
// the final density tree as BlendDensity divergence.
static float evalSplineD3(int idx, float c0, float c1, float c2, float c3,
    __global const int* sh, __global const float* fp, __global const int* ci)
{
    int b=idx*4, type=sh[b], K=sh[b+1], fpS=sh[b+2];
    if (type == 0) return fp[fpS];
    if (type == 1) {
        int vS = sh[b+3];
        int lo = findKnot(c0, K, fpS, fp);
        if (lo < 0)    return fp[vS]      + fp[fpS+K]      * (c0 - fp[fpS]);
        if (lo >= K-1) return fp[vS+K-1]  + fp[fpS+2*K-1] * (c0 - fp[fpS+K-1]);
        return hermiteInterp(c0, lo, K, fpS, fp[vS+lo], fp[vS+lo+1], fp);
    }
    int csS = sh[b+3];
    int lo  = findKnot(c0, K, fpS, fp);
    float vLo, vHi;
    if (lo < 0) {
        vLo = evalSplineD2(ci[csS], c1, c2, c3, sh, fp, ci);
        return vLo + fp[fpS+K] * (c0 - fp[fpS]);
    }
    if (lo >= K-1) {
        vLo = evalSplineD2(ci[csS+K-1], c1, c2, c3, sh, fp, ci);
        return vLo + fp[fpS+2*K-1] * (c0 - fp[fpS+K-1]);
    }
    vLo = evalSplineD2(ci[csS+lo],   c1, c2, c3, sh, fp, ci);
    vHi = evalSplineD2(ci[csS+lo+1], c1, c2, c3, sh, fp, ci);
    return hermiteInterp(c0, lo, K, fpS, vLo, vHi, fp);
}

// ═══════════════════════════════════════════════════════════════════
//  SECTION 3 — Stack machine opcodes
// ═══════════════════════════════════════════════════════════════════

#define OP_PUSH_CONST         0
#define OP_PUSH_X             1
#define OP_PUSH_Y             2
#define OP_PUSH_Z             3
#define OP_ADD                4
#define OP_MUL                5
#define OP_MIN_OP             6
#define OP_MAX_OP             7
#define OP_ABS                8
#define OP_SQUARE             9
#define OP_CUBE              10
#define OP_HALF_NEGATIVE     11
#define OP_QUARTER_NEGATIVE  12
#define OP_SQUEEZE           13
#define OP_INVERT            14
#define OP_CLAMP             15
#define OP_NOISE             16
#define OP_SHIFTED_NOISE     17
#define OP_Y_GRADIENT        18
#define OP_RANGE_SELECT      19
#define OP_BLEND_DENSITY_NOOP 20
#define OP_SPLINE_EVAL       21
#define OP_ADD_CONST         22
#define OP_MUL_CONST         23
#define OP_HALT             255

#define STACK_SIZE 64
#define MAX_ITERS 4096

#define OP_SHIFT_B_NOISE     24
#define OP_BLENDED_NOISE     25

#define OP_WEIRD_SCALED_SAMPLER 26

// ═══════════════════════════════════════════════════════════════════
//  SECTION 4 — Kernels
// ═══════════════════════════════════════════════════════════════════

__kernel void evalDensityTree(
    __global const double* positions,
    __global const int*    iOps,
    __global const double* dArgs,
    __global const double* noiseParams,
    __global const int*    noiseInfo,
    __global const double* octaveParams,
    __global const uchar*  permTables,
    __global const int*    splineHeaders,
    __global const float*  splineFloatPool,
    __global const int*    splineChildren,
    __global const double* blendedScalars,
    __global const double* blendedPerlinFactors,
    __global const int*    blendedPerlinInfo,
    __global       double* output,
    int count)
{
    int id = get_global_id(0);
    if (id >= count) return;

    double x = positions[id*3+0];
    double y = positions[id*3+1];
    double z = positions[id*3+2];

    double stack[STACK_SIZE];
    int sp = 0, ip = 0, dp = 0;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        int op = iOps[ip++];

        if (op == OP_HALT) break;

        switch (op) {
        case OP_PUSH_CONST: stack[sp++] = dArgs[dp++]; break;
        case OP_PUSH_X:     stack[sp++] = x; break;
        case OP_PUSH_Y:     stack[sp++] = y; break;
        case OP_PUSH_Z:     stack[sp++] = z; break;

        case OP_ADD: { double b=stack[--sp]; stack[sp-1]+=b; break; }
        case OP_MUL: { double b=stack[--sp]; stack[sp-1]*=b; break; }
        case OP_MIN_OP: { double b=stack[--sp]; stack[sp-1]=fmin(stack[sp-1],b); break; }
        case OP_MAX_OP: { double b=stack[--sp]; stack[sp-1]=fmax(stack[sp-1],b); break; }

        case OP_ADD_CONST: stack[sp-1] += dArgs[dp++]; break;
        case OP_MUL_CONST: stack[sp-1] *= dArgs[dp++]; break;

        case OP_ABS:              stack[sp-1]=fabs(stack[sp-1]); break;
        case OP_SQUARE:         { double v=stack[sp-1]; stack[sp-1]=v*v; break; }
        case OP_CUBE:           { double v=stack[sp-1]; stack[sp-1]=v*v*v; break; }
        case OP_HALF_NEGATIVE:  { double v=stack[sp-1]; stack[sp-1]= v>0.0?v:v*0.5; break; }
        case OP_QUARTER_NEGATIVE:{double v=stack[sp-1]; stack[sp-1]= v>0.0?v:v*0.25;break; }
        case OP_SQUEEZE: {
            double v=clamp(stack[sp-1],-1.0,1.0);
            stack[sp-1]=v/2.0-v*v*v/24.0; break;
        }
        case OP_INVERT: stack[sp-1]=1.0/stack[sp-1]; break;

        case OP_CLAMP: {
            double mn=dArgs[dp++], mx=dArgs[dp++];
            stack[sp-1]=clamp(stack[sp-1],mn,mx); break;
        }

        case OP_NOISE: {
            int ni=(int)iOps[ip++];
            double xzS=dArgs[dp++], yS=dArgs[dp++];
            stack[sp++]=evalNoise(ni,x*xzS,y*yS,z*xzS,
                noiseParams,noiseInfo,octaveParams,permTables);
            break;
        }

        case OP_SHIFTED_NOISE: {
            int ni=(int)iOps[ip++];
            double xzS=dArgs[dp++], yS=dArgs[dp++];
            double sz=stack[--sp], sy=stack[--sp], sx=stack[--sp];
            stack[sp++]=evalNoise(ni,x*xzS+sx,y*yS+sy,z*xzS+sz,
                noiseParams,noiseInfo,octaveParams,permTables);
            break;
        }

        case OP_SHIFT_B_NOISE: {
            int ni=(int)iOps[ip++];
            double xzS=dArgs[dp++];
            double sz=stack[--sp], sy=stack[--sp], sx=stack[--sp];
            // ShiftB: getValue(z*xzS + sx, x*xzS + sy, 0 + sz)
            // matches Java: offsetNoise.getValue(z*0.25, x*0.25, 0.0)
            stack[sp++]=evalNoise(ni, z*xzS+sx, x*xzS+sy, 0.0+sz,
                noiseParams,noiseInfo,octaveParams,permTables);
            break;
        }

        case OP_BLENDED_NOISE: {
            int bi = iOps[ip++];
            if (sp >= STACK_SIZE) { output[id] = 0.0; return; }
            stack[sp++] = evalBlendedNoise(bi, x, y, z,
                blendedScalars, blendedPerlinFactors, blendedPerlinInfo,
                octaveParams, permTables);
            break;
        }

        case OP_WEIRD_SCALED_SAMPLER: {
            int ni         = iOps[ip++];
            int mapperType = iOps[ip++];
            double input   = stack[--sp];
            double d = (mapperType == 0)
                ? getSpaghettiRarity3D(input)
                : getSphaghettiRarity2D(input);
            // d * abs(noise.getValue(x/d, y/d, z/d))
            double nx = evalNoise(ni, x/d, y/d, z/d,
                noiseParams, noiseInfo, octaveParams, permTables);
            stack[sp++] = d * fabs(nx);
            break;
        }

        case OP_Y_GRADIENT: {
            int fy=iOps[ip++], ty=iOps[ip++];
            double fv=dArgs[dp++], tv=dArgs[dp++];
            double t=clamp((y-(double)fy)/((double)ty-(double)fy),0.0,1.0);
            stack[sp++]=fv+t*(tv-fv); break;
        }

        case OP_RANGE_SELECT: {
            double mn=dArgs[dp++], mx=dArgs[dp++];
            double outR=stack[--sp], inR=stack[--sp], inp=stack[--sp];
            stack[sp++]=(inp>=mn && inp<mx) ? inR : outR; break;
        }

        case OP_BLEND_DENSITY_NOOP:
            // identity: wrapped function result is already on stack
            break;

        case OP_SPLINE_EVAL: {
            int si    = iOps[ip++];
            int depth = iOps[ip++];

            // Pop coords: c0=outermost (top of stack), c1, c2, c3 (deepest)
            float c0 = (depth >= 1) ? (float)stack[--sp] : 0.0f;
            float c1 = (depth >= 2) ? (float)stack[--sp] : 0.0f;
            float c2 = (depth >= 3) ? (float)stack[--sp] : 0.0f;
            float c3 = (depth >= 4) ? (float)stack[--sp] : 0.0f;
            // Drain any deeper coords (depth > 4) — not yet supported,
            // the result will be wrong but stack stays balanced.
            for (int extra = 4; extra < depth; extra++) --sp;

            float r;
            if      (depth <= 1) r = evalSplineD0(si, c0, splineHeaders, splineFloatPool, splineChildren);
            else if (depth == 2) r = evalSplineD1(si, c0, c1, splineHeaders, splineFloatPool, splineChildren);
            else if (depth == 3) r = evalSplineD2(si, c0, c1, c2, splineHeaders, splineFloatPool, splineChildren);
            else                 r = evalSplineD3(si, c0, c1, c2, c3, splineHeaders, splineFloatPool, splineChildren);

            stack[sp++] = (double)r;
            break;
        }
        } // switch
    } // for iter

    output[id] = (sp > 0) ? stack[sp-1] : 0.0;
}

// ── Original single-octave kernel (kept for validation) ──────────────────────
__kernel void sampleNoise(
    __global const double* positions, __global const uchar* perm,
    double xo, double yo, double zo, __global double* output, int count)
{
    int id = get_global_id(0); if (id >= count) return;
    output[id] = improvedNoise(positions[id*3], positions[id*3+1], positions[id*3+2], xo,yo,zo, perm);
}

// ── NormalNoise batch kernel (kept for validation) ────────────────────────────
#define NORMAL_NOISE_INPUT_FACTOR 1.0181268882175227
__kernel void sampleNormalNoise(
    __global const double* positions, __global const double* noiseP,
    __global const double* octP, __global const uchar* perms,
    __global double* output, int firstN, int secondN, int count)
{
    int id = get_global_id(0); if (id >= count) return;
    double x=positions[id*3], y=positions[id*3+1], z=positions[id*3+2];
    double vf=noiseP[0], fiF=noiseP[1], fvF=noiseP[2], siF=noiseP[3], svF=noiseP[4];
    double fv=perlinGetValue(x,y,z,fiF,fvF,firstN,0,octP,perms);
    double sv=perlinGetValue(x*NORMAL_NOISE_INPUT_FACTOR,y*NORMAL_NOISE_INPUT_FACTOR,
                              z*NORMAL_NOISE_INPUT_FACTOR,siF,svF,secondN,firstN,octP,perms);
    output[id]=(fv+sv)*vf;
}
