package io.papermc.paper.gpu;

/** Opcodes for the GlassPaper density function stack machine. */
public final class DfOpcode {
    private DfOpcode() {}

    // ── Stack / coordinates ──────────────────────────────────────────────────
    public static final int PUSH_CONST         = 0;   // dArgs: [value]
    public static final int PUSH_X             = 1;
    public static final int PUSH_Y             = 2;
    public static final int PUSH_Z             = 3;

    // ── Binary arithmetic ────────────────────────────────────────────────────
    public static final int ADD                = 4;
    public static final int MUL                = 5;
    public static final int MIN_OP             = 6;
    public static final int MAX_OP             = 7;

    // ── Unary transforms (Mapped) ────────────────────────────────────────────
    public static final int ABS                = 8;
    public static final int SQUARE             = 9;
    public static final int CUBE               = 10;
    public static final int HALF_NEGATIVE      = 11;
    public static final int QUARTER_NEGATIVE   = 12;
    public static final int SQUEEZE            = 13;
    public static final int INVERT             = 14;

    // ── Clamp ────────────────────────────────────────────────────────────────
    public static final int CLAMP              = 15;  // dArgs: [min, max]

    // ── Noise ────────────────────────────────────────────────────────────────
    public static final int NOISE              = 16;  // iArgs: [noise_idx], dArgs: [xzScale, yScale]
    public static final int SHIFTED_NOISE      = 17;  // iArgs: [noise_idx], dArgs: [xzScale, yScale]
    // pops [shiftX, shiftY, shiftZ] from stack first

    // ── Y-clamped gradient ───────────────────────────────────────────────────
    public static final int Y_GRADIENT         = 18;  // iArgs: [fromY, toY], dArgs: [fromVal, toVal]

    // ── Range choice ─────────────────────────────────────────────────────────
    // Stack before:  [..., input, in_range_result, out_range_result]
    // Stack after:   [..., selected_result]
    public static final int RANGE_SELECT       = 19;  // dArgs: [minInclusive, maxExclusive]

    // ── Blend density (no-op for empty blender) ──────────────────────────────
    public static final int BLEND_DENSITY_NOOP = 20;

    // ── Spline ───────────────────────────────────────────────────────────────
    // Before this opcode, coords are pushed deepest-first:
    //   push coord[depth-1] (innermost)  ...  push coord[0] (outermost, top of stack)
    // iArgs: [spline_idx, depth]
    public static final int SPLINE_EVAL        = 21;

    // ── MulOrAdd shortcuts ───────────────────────────────────────────────────
    public static final int ADD_CONST          = 22;  // dArgs: [constant]
    public static final int MUL_CONST          = 23;  // dArgs: [constant]

    public static final int SHIFT_B_NOISE = 24; // like SHIFTED_NOISE but swaps X/Z axes
    public static final int BLENDED_NOISE = 25; // iArgs: [blendedNoiseIdx]

    public static final int WEIRD_SCALED_SAMPLER = 26; // iArgs: [noise_idx, mapper_type]
    // mapper_type: 0=TYPE1(3D), 1=TYPE2(2D)
    // pops input value from stack first

    // ── Sentinel ─────────────────────────────────────────────────────────────
    public static final int HALT               = 255;
}
