// {% copyright %}

#include <math.h>
#include <param_postops.h>

#include <dma_shave.h>
#include <sw_shave_lib_common.h>
#include <moviVectorConvert.h>

using nn::shave_lib::PostOpsParams;
using nn::shave_lib::t_ClampLayerParams;
using nn::shave_lib::t_PowerLayerParams;
using nn::shave_lib::t_PostOps;
using nn::shave_lib::t_CHWPostOps3DParams;
using nn::shave_lib::t_HCWPostOps3DParams;
using nn::shave_lib::t_HWCPostOps3DParams;
using namespace subspace;

extern "C"
{
void chw_postOps_3D_core(nn::shave_lib::t_CHWPostOps3DParams *params);
void hwc_postOps_3D_core(nn::shave_lib::t_HWCPostOps3DParams *params);
void hcw_postOps_3D_core(nn::shave_lib::t_HCWPostOps3DParams *params);
}

#define MIN(_a, _b) __builtin_shave_cmu_min_i32_rr_int((_a), (_b))
#define UNROLL_SIZE           8 // Changes to this should be reflected in the code as well.
const half X_MAX = std::numeric_limits<half>::max();

typedef void (*operation_chw_type)(half8*, half8*, half*, half*, s32, s32, half, void*);
typedef void (*operation_hwc_type)(half8*, half8*, half8*, half8*, s32, s32, half, void*);
typedef void (*operation_hcw_type)(half8*, half8*, half8*, half8*, s32, s32, half, void*);

#define intrinsic_vec(intrinsic, vin, vout) \
        (vout)[0] = intrinsic((vin)[0]); \
        (vout)[1] = intrinsic((vin)[1]); \
        (vout)[2] = intrinsic((vin)[2]); \
        (vout)[3] = intrinsic((vin)[3]); \
        (vout)[4] = intrinsic((vin)[4]); \
        (vout)[5] = intrinsic((vin)[5]); \
        (vout)[6] = intrinsic((vin)[6]); \
        (vout)[7] = intrinsic((vin)[7]);

#define log2_vec(vin, vout) intrinsic_vec(__builtin_shave_sau_log2_f16_l_r, vin, vout)
#define exp2_vec(vin, vout) intrinsic_vec(__builtin_shave_sau_exp2_f16_l_r, vin, vout)

#define FP16_BIAS 15 /* expo bias */
#define FP16_TOTALBITS 16 /* total number of bits */
#define FP16_FRACTBITS 10 /* number of explicit fraction bits */
#define FP16_GREATINT ((FP16_BIAS + FP16_FRACTBITS) << FP16_FRACTBITS) /* big: all equal or above are integers */
#define FP16_TRUNCFRACT (((unsigned)-1 << FP16_FRACTBITS)) /* mask to truncate fraction bits at the binary point */

const int fp32_bias = 127;
const int fp32_sigbits = 23; // significand length, in bits (hidden leading '1' not counted)
const int fp32_sigmask = (1 << fp32_sigbits) - 1; // significand field mask

static inline float8 addf8(float8 a, float8 b)
{
    float8 res;
    res.lo = __builtin_shave_vau_add_f32_rr(a.lo, b.lo);
    res.hi = __builtin_shave_vau_add_f32_rr(a.hi, b.hi);
    return res;
}

static inline float8 subf8(float8 a, float8 b)
{
    float8 res;
    res.lo = __builtin_shave_vau_sub_f32_rr(a.lo, b.lo);
    res.hi = __builtin_shave_vau_sub_f32_rr(a.hi, b.hi);
    return res;
}

// exponential function approximation
// Taylor series expansion used: exp(x) ~ 1 + x/1! + x^2/2! + x^3/3! + ...
// for FP16 result, x^2 term is enough; for more precision, add more expansion terms
// with true minimax polynomial approximation we can get slightly more precise result
static inline float8 expf8(float8 x)
{
    const float8 invln2  = (float8) (int8)0x3fb8aa3b; //  1/log(2) = 1.44269502162933349609375

    const float8 ln2hi   = (float8) (int8)0x3f317200; // log(2) hi = 0.693145751953125
    const float8 ln2lo   = (float8) (int8)0x35bfbe8e; // log(2) lo = 1.42860676533018704503775e-06

    const float8 rsh     = (float8) (int8)0x4b400000; // 1.5*2^23 = 12582912 (fp32 round-to-nearest shifter)

    const float8 p2 = (float8)(1.0f / 2); // 1/2!

    const float8 x_div_ln2 = (x * invln2);

    const float8 tmp = addf8(x_div_ln2, rsh);
    const float8 scale = (float8)(((uint8)tmp + fp32_bias) << fp32_sigbits);
    const float8 n = (tmp - rsh);

    const float8 r = subf8(x, n * ln2hi) - (n * ln2lo);
    const float8 poly = p2 * r * r + r;

    const float8 res = poly * scale + scale;

    return res;
}

// natural logarithmic function approximation
// Taylor series expansion used: log(1 + x) ~ x - x^2/2 + x^3/3 - x^4/4 + ...
// for FP16 result, x^4 term is enough; for more precision, add more expansion terms
// with true minimax polynomial approximation we can get slightly more precise result
static inline float8 logf8(float8 x)
{
    const int8 brk = (int8)0x3f2aaaab; // 2/3 = 0.666666686534881591796875

    const float8 one = (float8)1.0f;
    const float8 log2 = (float8) (int8)0x3f317218; // log(2) = 0.693147182464599609375

    const float8 p2 = (float8)(-1.0f / 2);
    const float8 p3 = (float8)( 1.0f / 3);
    const float8 p4 = (float8)(-1.0f / 4);

    const int8 ix = (int8)x - brk;
    const int8 in = (ix >> fp32_sigbits);
    const int8 ir = (ix & fp32_sigmask) + brk;

    const float8 n = mvuConvert_float8(in);
    const float8 r = (float8)ir - one;

    const float8 poly = (((p4 * r + p3) * r) + p2) * r * r + r;

    const float8 res = n * log2 + poly;

    return res;
}

/* vector implementation of library ceilf(x) function for half floats */
static inline half8 ceilh8(half8 x)
{
    const short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
    const half8 ones = (half8)1.0f;

    short8 xInt = (short8)x;
    short8 xSign = (xInt & signMask);
    short8 xAbs = (xInt & ~signMask);

    short8 xExpo = (xAbs >> FP16_FRACTBITS) - (short8)FP16_BIAS;
    short8 truncMask = ((short8)FP16_TRUNCFRACT >> xExpo);

    short8 isSign = (xInt >> (FP16_TOTALBITS - 1));
    short8 isZero = (xAbs - (short8)1) >> (FP16_TOTALBITS - 1);
    short8 isSmall = (xAbs - (short8)ones) >> (FP16_TOTALBITS - 1);
    isSmall &= ~isZero;
    short8 isGreat = ((short8)(FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1);
    short8 isExact = (isGreat | isZero);
    short8 isCeil = ~(isExact | isSmall);

    short8 xTrunc = (xAbs & truncMask);
    short8 isInexact = (short8)(xTrunc != xAbs);

    short8 xCeil = (short8)((half8)xTrunc + (half8)((~isSign & isInexact) & (short8)ones));
    short8 xSmall = (short8)((~isSign & isSmall) & (short8)ones);

    half8 res = (half8)( (isExact & xInt) | (xSign | ((isCeil & xCeil) | (isSmall & xSmall))) );

    return res;
}

/* vector implementation of library floorf(x) function for half floats */
static inline half8 floorh8(half8 x)
{
    const short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
    const half8 ones = (half8)1.0f;

    short8 xInt = (short8)x;
    short8 xSign = (xInt & signMask);
    short8 xAbs = (xInt & ~signMask);

    short8 xExpo = (xAbs >> FP16_FRACTBITS) - (short8)FP16_BIAS;
    short8 truncMask = ((short8)FP16_TRUNCFRACT >> xExpo);

    short8 isSign = (xInt >> (FP16_TOTALBITS - 1));
    short8 isZero = (xAbs - (short8)1) >> (FP16_TOTALBITS - 1);
    short8 isSmall = (xAbs - (short8)ones) >> (FP16_TOTALBITS - 1);
    isSmall &= ~isZero;
    short8 isGreat = ((short8)(FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1);
    short8 isExact = (isGreat | isZero);
    short8 isFloor = ~(isExact | isSmall);

    short8 xTrunc = (xAbs & truncMask);
    short8 isInexact = (short8)(xTrunc != xAbs);

    short8 xFloor = (short8)((half8)xTrunc + (half8)((isSign & isInexact) & (short8)ones));
    short8 xSmall = (short8)((isSign & isSmall) & (short8)ones);

    half8 res = (half8)( (isExact & xInt) | (xSign | ((isFloor & xFloor) | (isSmall & xSmall))) );

    return res;
}

/* vector implementation of library roundf(x) function for half floats */
static inline half8 roundh8(half8 x, nn::shave_lib::roundMode mode)
{
    // the compiler moves this condition outside the cycle automatically
    if (mode == nn::shave_lib::roundMode::HALF_TO_EVEN)
    {
        short8 xInt = (short8)x;
        short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
        short8 xSign = (xInt & signMask);
        short8 xAbs = (xInt & ~signMask);

        short8 isGreat = ((short8)(FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1);

        half8 roundShift = (half8)((short8)FP16_GREATINT);
        half8 vround = __builtin_shave_vau_sub_f16_rr(((half8)xAbs + roundShift), roundShift);
        half8 xres = (half8)((~isGreat & (short8)vround) | ((isGreat) & xInt) | xSign);

        return xres;
    }
    else // mode == nn::shave_lib::roundMode::HALF_AWAY_FROM_ZERO
    {
        const short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
        const half8 halves = (half8)0.5f;

        short8 xInt = (short8)x;

        short8 xSign = (xInt & signMask);
        short8 xAbs = (xInt & ~signMask);

        short8 xAbsHalfPlus = (short8)((half8)(xAbs) + half8(0.5f));

        short8 xExpo = (xAbsHalfPlus >> FP16_FRACTBITS) - (short8)FP16_BIAS;
        short8 truncMask = ((short8)FP16_TRUNCFRACT >> xExpo);

        short8 isSmall = (xAbs - (short8)halves) >> (FP16_TOTALBITS - 1);
        short8 isGreat = ((short8)(FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1);
        short8 isTrunc = ~(isGreat | isSmall);

        short8 xTrunc = (xAbsHalfPlus & truncMask);
        half8 res = (half8)((isGreat & xInt) | xSign | (isTrunc & xTrunc));

        return res;
    }
}

static void  ceil_fp16(const half8 * __restrict__    in,
                             half8 * __restrict__    out,
                       const half8 * __restrict__ /* weights */,
                       const half8 * __restrict__ /* biases */,
                       const int                     no_lines,
                       const int                     line_size,
                       const half                 /* x */,
                       const void  *              /* parameters */)
{
    const int tile_size = no_lines * line_size;

    #pragma clang loop unroll_count(4)
    for(int i = 0; i < tile_size; ++i)
    {
        out[i] = ceilh8(in[i]);
    }
}

/// @todo: investigate possibilities to get rid of float type.
static inline half8 erfh8(half8 x)
{
    const short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
    const half8 clipBound = (half8)2.86f;

    const half8 xAbs = (half8)(~signMask & (short8)x);
    const short8 xSign = (signMask & (short8)x);

    //  Points clip_bound and -clip_bound are extremums for this polynom
    //  So in order to provide better accuracy comparing to std::erf we have to clip input range
    const short8 isBig = (xAbs > clipBound); // return +-1;
    const short8 inRange = ~isBig;

    const half8 one = (half8)1.0f;
    const half8 big = (half8)(xSign | (short8)one);

    //  A polynomial approximation of the error function
    const float8 erfNumerator[4] = { (float8)90.0260162353515625f, (float8)2232.00537109375f,
                                     (float8)7003.3251953125f, (float8)55592.30078125f };
    const float8 erfDenominator[5] = { (float8)33.56171417236328125f, (float8)521.35797119140625f,
                                       (float8)4594.32373046875f, (float8)22629.0f, (float8)49267.39453125f };

    const float8 xf = mvuConvert_float8(x);
    const float8 x2 = xf * xf;

    float8 num = (float8)9.60497379302978515625f;
    for (const auto c : erfNumerator)
        num = num * x2 + c;
    num *= xf;

    float8 den = (float8)1.0f;
    for (const auto c : erfDenominator)
        den = den * x2 + c;

    half8 res = mvuConvert_half8(num / den);

    return (half8)((isBig & (short8)big) | (inRange & (short8)res));
}

static void floor_fp16(const half8 * __restrict__    in,
                             half8 * __restrict__    out,
                       const half8 * __restrict__ /* weights */,
                       const half8 * __restrict__ /* biases */,
                       const int                     no_lines,
                       const int                     line_size,
                       const half                 /* x */,
                       const void  *              /* parameters */)
{
    const int tile_size = no_lines * line_size;

    #pragma clang loop unroll_count(4)
    for(int i = 0; i < tile_size; ++i)
    {
        out[i] = floorh8(in[i]);
    }
}

void round_fp16(const half8 * __restrict__    in,
                      half8 * __restrict__    out,
                const half8 * __restrict__ /* weights */,
                const half8 * __restrict__ /* biases */,
                const int                     no_lines,
                const int                     line_size,
                const half                 /* x */,
                const void  *                 parameters)
{
    using nn::shave_lib::t_RoundLayerParams;
    const auto* roundParams = reinterpret_cast<const t_RoundLayerParams*>(parameters);
    const auto mode = roundParams->mode;

    const int tile_size = no_lines * line_size;

    #pragma clang loop unroll_count(4)
    for(int i = 0; i < tile_size; ++i)
    {
        out[i] = roundh8(in[i], mode);
    }
}

static void erf_fp16(const half8 * __restrict__    in,
                           half8 * __restrict__    out,
                     const half8 * __restrict__ /* weights */,
                     const half8 * __restrict__ /* biases */,
                     const int                     no_lines,
                     const int                     line_size,
                     const half                 /* x */,
                     const void  *              /* parameters */)
{
    const int tile_size = no_lines * line_size;

    #pragma clang loop unroll_count(4)
    for(int i = 0; i < tile_size; ++i)
    {
        out[i] = erfh8(in[i]);
    }
}

static void gelu_fp16(const half8* __restrict__  in,
                            half8* __restrict__  out,
                      const half8* __restrict__  /* weights */,
                      const half8* __restrict__  /* biases */,
                      const int                  no_lines,
                      const int                  line_size,
                      const half                 /* x */,
                      const void *               /*parameters */)
{
    // Gelu(x) = x * P(X <=x) = x * F(x) = 0.5 * x * (1 + erf(x / sqrt(2))
    // which in turn is approximated to
    // 0.5 * x * ( 1 + tanh[sqrt(2 / pi) * (x + 0.044715 * x^3)])
    // == 0.5 * x * (2 * exp(c) / (exp(c) + exp(-c)), where c = sqrt(2 / pi) * (x + 0.044715 * x^3)
    // == x / (1 + exp(-2 * sqrt(2 / pi) * x * (1 + 0.044715 * x^2)))

    const int numVectors = no_lines * line_size;
    const half8 one = (half8)1.0f;
    const half8 sqrt_8_divide_by_pi = (half8)-1.59576912161f; // -2 * sqrt(2 / pi)
    const half8 fitting_const = (half8)-0.07135481627f; // -2 * sqrt(2 / pi) * 0.044715

    const uint16_t inv_ln2 = 0x3dc5;
    const half inv_ln2_h = *(const half*)&inv_ln2;

#pragma clang loop unroll_count(4)
    for(s32 i = 0; i < numVectors; ++i)
    {
        half8 x = in[i];
        half8 tmp = x * (sqrt_8_divide_by_pi + fitting_const * x * x);

        half8 exp_x = tmp * inv_ln2_h;
        exp2_vec(exp_x, exp_x);

        out[i] = x / (one + exp_x);
    }
}

void log_fp16(const half8* __restrict__ in,
                    half8* __restrict__ out,
              const half8* __restrict__ /* weights */,
              const half8* __restrict__ /* biases */,
              const int                 no_lines,
              const int                 line_size,
              const half                /* x */,
              const void *              /*parameters*/)
{
    const unsigned short ln2 = 0x398c;
    const half ln2_h = *reinterpret_cast<const half *>(&ln2);

    const int numVectors = no_lines * line_size;

    #pragma clang loop unroll_count(8)
    for(s32 i = 0; i < numVectors; ++i)
    {
        log2_vec(in[i], out[i]);
    }

    #pragma clang loop unroll_count(8)
    for(s32 i = 0; i < numVectors; ++i)
    {
        out[i] = out[i] * ln2_h;
    }
}

void exp_fp16(const half8 * __restrict__ in,
                    half8 * __restrict__ out,
              const half8 * __restrict__ /* weights */,
              const half8 * __restrict__ /* biases */,
              const int                  no_lines,
              const int                  line_size,
              const half                 /* x */,
              const void  *              /* parameters */)
{
    const uint16_t inv_ln2 = 0x3dc5;
    const half inv_ln2_h = *(const half*)&inv_ln2;

    const int tile_size = no_lines * line_size;

    #pragma clang loop unroll_count(4)
    for(s32 i = 0; i < tile_size; ++i)
    {
        half8 exp_x = in[i] * inv_ln2_h;
        exp2_vec(exp_x, exp_x);
        out[i] = exp_x;
    }
}

void HWC_biasRelu_fp16(half8 * __restrict__ data_in,
                       half8 * __restrict__ data_out,
                       half8 * __restrict__ weights,
                       half8 * __restrict__ bias,
                       s32 no_lines, s32 line_size, half x,
                       void *parameters)
{
    UNUSED(weights);
    UNUSED(parameters);

    const half8  x_vec = (x <= half(0.0)) ? (half8)(X_MAX) : (half8)x;

    s32 line_i = 0;

    for (; line_i < line_size - 3; line_i += 4)
    {
        half8 rb0 = bias[line_i + 0];
        half8 rb1 = bias[line_i + 1];
        half8 rb2 = bias[line_i + 2];
        half8 rb3 = bias[line_i + 3];

        half8 * __restrict__ rin = &data_in[line_i];
        half8 * __restrict__ rout = &data_out[line_i];

        s32 i = 0;

        if (no_lines > 3)
        {
            half8 rr00 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[0] + rb0, x_vec);
            half8 rr01 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[1] + rb1, x_vec);
            half8 rr02 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[2] + rb2, x_vec);
            half8 rr03 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[3] + rb3, x_vec);
            rin += line_size;

            half8 rr10 = rin[0] + rb0;
            half8 rr11 = rin[1] + rb1;
            half8 rr12 = rin[2] + rb2;
            half8 rr13 = rin[3] + rb3;
            rin += line_size;

            half8 rr20 = rin[0];
            half8 rr21 = rin[1];
            half8 rr22 = rin[2];
            half8 rr23 = rin[3];
            rin += line_size;

#pragma clang loop unroll_count(4)
            for (; i < no_lines - 3; ++i)
            {
                half8 rr30 = rin[0];
                half8 rr31 = rin[1];
                half8 rr32 = rin[2];
                half8 rr33 = rin[3];
                rin += line_size;

                rout[0] = rr00;
                rout[1] = rr01;
                rout[2] = rr02;
                rout[3] = rr03;
                rout += line_size;

                rr00 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr10, x_vec);
                rr01 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr11, x_vec);
                rr02 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr12, x_vec);
                rr03 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr13, x_vec);

                rr10 = rr20 + rb0;
                rr11 = rr21 + rb1;
                rr12 = rr22 + rb2;
                rr13 = rr23 + rb3;

                rr20 = rr30;
                rr21 = rr31;
                rr22 = rr32;
                rr23 = rr33;
            }

            rin -= 3 * line_size;
        }

        for (; i < no_lines; ++i)
        {
            rout[0] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[0] + rb0, x_vec);
            rout[1] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[1] + rb1, x_vec);
            rout[2] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[2] + rb2, x_vec);
            rout[3] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[3] + rb3, x_vec);
            rin += line_size;
            rout += line_size;
        }
    }

    for (; line_i < line_size; ++line_i)
    {
        half8 rb0 = bias[line_i];

        half8 * __restrict__ rin = &data_in[line_i];
        half8 * __restrict__ rout = &data_out[line_i];

        s32 i = 0;

        if (no_lines > 3)
        {
            half8 rr0 = rin[0];
            rin += line_size;
            rr0 = rr0 + rb0;
            rr0 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr0, x_vec);

            half8 rr1 = rin[0];
            rin += line_size;
            rr1 = rr1 + rb0;

            half8 rr2 = rin[0];
            rin += line_size;

#pragma clang loop unroll_count(8)
            for (; i < no_lines - 3; ++i)
            {
                half8 rr3 = rin[0];
                rin += line_size;

                rout[0] = rr0;
                rout += line_size;

                rr0 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr1, x_vec);

                rr1 = rr2 + rb0;

                rr2 = rr3;
            }

            rin -= 3 * line_size;
        }

        for (; i < no_lines; ++i)
        {
            rout[0] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[0] + rb0, x_vec);
            rin += line_size;
            rout += line_size;
        }
    }
}

void HWC_biasLeakyRelu_fp16(half8 * __restrict__ data_in,
                            half8 * __restrict__ data_out,
                            half8 * __restrict__ weights,
                            half8 * __restrict__ bias,
                            s32 no_lines, s32 line_size, half x,
                            void *parameters)
{
    UNUSED(weights);
    UNUSED(parameters);

    const half8 zeros = (half8)0.0;
    const half8 x_vec = (half8)x;

    s32 line_i = 0;

    for (; line_i < line_size - 1; line_i += 2)
    {
        half8 rb00 = bias[line_i + 0];
        half8 rb01 = bias[line_i + 1];

        half8 * __restrict__ rin = &data_in[line_i];
        half8 * __restrict__ rout = &data_out[line_i];

        s32 i = 0;

        if (no_lines > 4)
        {
            half8 rr00 = rin[0] + rb00;
            half8 rr01 = rin[1] + rb01;
            rin += line_size;
            rr00 = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            rr01 = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);

            half8 rr10 = rin[0] + rb00;
            half8 rr11 = rin[1] + rb01;
            rin += line_size;
            half8 rq10 = __builtin_shave_cmu_min_f16_rr_half8(rr10, zeros); rr10 = __builtin_shave_cmu_max_f16_rr_half8(rr10, zeros);
            half8 rq11 = __builtin_shave_cmu_min_f16_rr_half8(rr11, zeros); rr11 = __builtin_shave_cmu_max_f16_rr_half8(rr11, zeros);

            half8 rr20 = rin[0] + rb00;
            half8 rr21 = rin[1] + rb01;
            rin += line_size;

            half8 rr30 = rin[0];
            half8 rr31 = rin[1];
            rin += line_size;

#pragma clang loop unroll_count(4)
            for (; i < no_lines - 4; ++i)
            {
                half8 rr40 = rin[0];
                half8 rr41 = rin[1];
                rin += line_size;

                rout[0] = rr00;
                rout[1] = rr01;
                rout += line_size;

                rr00 = rr10 + rq10 * x_vec;
                rr01 = rr11 + rq11 * x_vec;

                rq10 = __builtin_shave_cmu_min_f16_rr_half8(rr20, zeros); rr10 = __builtin_shave_cmu_max_f16_rr_half8(rr20, zeros);
                rq11 = __builtin_shave_cmu_min_f16_rr_half8(rr21, zeros); rr11 = __builtin_shave_cmu_max_f16_rr_half8(rr21, zeros);

                rr20 = rr30 + rb00;
                rr21 = rr31 + rb01;

                rr30 = rr40;
                rr31 = rr41;
            }

            rin -= 4 * line_size;
        }

        for (; i < no_lines; ++i)
        {
            half8 rr00 = rin[0] + rb00;
            half8 rr01 = rin[1] + rb01;
            rout[0] = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            rout[1] = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);
            rin += line_size;
            rout += line_size;
        }
    }

    for (; line_i < line_size; ++line_i)
    {
        half8 rb0 = bias[line_i];

        half8 * __restrict__ rin = &data_in[line_i];
        half8 * __restrict__ rout = &data_out[line_i];

        s32 i = 0;

        if (no_lines > 4)
        {
            half8 rr0 = rin[0] + rb0;
            rin += line_size;
            half8 rq0 = __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros); rr0 = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros);
            rr0 = rr0 + rq0 * x_vec;

            half8 rr1 = rin[0] + rb0;
            rin += line_size;
            half8 rq1 = __builtin_shave_cmu_min_f16_rr_half8(rr1, zeros); rr1 = __builtin_shave_cmu_max_f16_rr_half8(rr1, zeros);

            half8 rr2 = rin[0] + rb0;
            rin += line_size;

            half8 rr3 = rin[0];
            rin += line_size;

#pragma clang loop unroll_count(7)
            for (; i < no_lines - 4; ++i)
            {
                half8 rr4 = rin[0];
                rin += line_size;

                rout[0] = rr0;
                rout += line_size;

                rr0 = rr1 + rq1 * x_vec;

                rq1 = __builtin_shave_cmu_min_f16_rr_half8(rr2, zeros); rr1 = __builtin_shave_cmu_max_f16_rr_half8(rr2, zeros);

                rr2 = rr3 + rb0;

                rr3 = rr4;
            }

            rin -= 4 * line_size;
        }

        for (; i < no_lines; ++i)
        {
            half8 rr0 = rin[0] + rb0;
            rout[0] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros);
            rin += line_size;
            rout += line_size;
        }
    }
}

void leakyRelu_fp16(half8 * __restrict__ data_in,
                    half8 * __restrict__ data_out,
                    half8 * __restrict__ weights,
                    half8 * __restrict__ bias,
                    s32 no_lines, s32 line_size, half x,
                    void *parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(parameters);

#if defined(MA2480)

    const half8 zeros = (half8)0.0;
        const half8 x_vec = (half8)x;

        s32 tile_size = line_size * no_lines;

        s32 i = 0;

        if (tile_size >= UNROLL_SIZE)
        {
            half8 r00 = data_in[0];
            half8 r01 = data_in[1];
            half8 r02 = data_in[2];
            half8 r03 = data_in[3];
            half8 r04 = data_in[4];
            half8 r05 = data_in[5];
            half8 r06 = data_in[6];
            half8 r07 = data_in[7];
            data_in += UNROLL_SIZE;

            half8 q00 = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + __builtin_shave_cmu_min_f16_rr_half8(r00, zeros) * x_vec;
            half8 q01 = __builtin_shave_cmu_min_f16_rr_half8(r01, zeros);
            half8 q02 = __builtin_shave_cmu_min_f16_rr_half8(r02, zeros);
            half8 q03 = __builtin_shave_cmu_min_f16_rr_half8(r03, zeros);
            half8 q04 = __builtin_shave_cmu_min_f16_rr_half8(r04, zeros);
            half8 q05 = __builtin_shave_cmu_min_f16_rr_half8(r05, zeros);
            half8 q06 = __builtin_shave_cmu_min_f16_rr_half8(r06, zeros);
            half8 q07 = __builtin_shave_cmu_min_f16_rr_half8(r07, zeros);

            for (i = UNROLL_SIZE; i < tile_size - (UNROLL_SIZE - 1); i += UNROLL_SIZE)
            {
                half8 r10 = data_in[0];
                half8 r11 = data_in[1];
                half8 r12 = data_in[2];
                half8 r13 = data_in[3];
                half8 r14 = data_in[4];
                half8 r15 = data_in[5];
                half8 r16 = data_in[6];
                half8 r17 = data_in[7];
                data_in += UNROLL_SIZE;

                half8 q10 = __builtin_shave_cmu_max_f16_rr_half8(r10, zeros) + __builtin_shave_cmu_min_f16_rr_half8(r10, zeros) * x_vec;
                half8 q11 = __builtin_shave_cmu_min_f16_rr_half8(r11, zeros);
                half8 q12 = __builtin_shave_cmu_min_f16_rr_half8(r12, zeros);
                half8 q13 = __builtin_shave_cmu_min_f16_rr_half8(r13, zeros);
                half8 q14 = __builtin_shave_cmu_min_f16_rr_half8(r14, zeros);
                half8 q15 = __builtin_shave_cmu_min_f16_rr_half8(r15, zeros);
                half8 q16 = __builtin_shave_cmu_min_f16_rr_half8(r16, zeros);
                half8 q17 = __builtin_shave_cmu_min_f16_rr_half8(r17, zeros);

                data_out[0] = q00;
                data_out[1] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + q01 * x_vec;
                data_out[2] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + q02 * x_vec;
                data_out[3] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + q03 * x_vec;
                data_out[4] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + q04 * x_vec;
                data_out[5] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + q05 * x_vec;
                data_out[6] = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + q06 * x_vec;
                data_out[7] = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + q07 * x_vec;
                data_out += UNROLL_SIZE;

                r00 = r10; q00 = q10;
                r01 = r11; q01 = q11;
                r02 = r12; q02 = q12;
                r03 = r13; q03 = q13;
                r04 = r14; q04 = q14;
                r05 = r15; q05 = q15;
                r06 = r16; q06 = q16;
                r07 = r17; q07 = q17;
            }

            data_out[0] = q00;
            data_out[1] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + q01 * x_vec;
            data_out[2] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + q02 * x_vec;
            data_out[3] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + q03 * x_vec;
            data_out[4] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + q04 * x_vec;
            data_out[5] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + q05 * x_vec;
            data_out[6] = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + q06 * x_vec;
            data_out[7] = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + q07 * x_vec;
            data_out += UNROLL_SIZE;
        }

        for (; i < tile_size; ++i)
        {
            half8 r = *data_in++;
            *data_out++ = __builtin_shave_cmu_max_f16_rr_half8(r, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(r, zeros);
        }

#else // MA2480

    const half8 zeros = (half8)0.0;
    const half8 x_vec = (half8)x;

    s32 tile_size = line_size * no_lines;

    s32 i = 0;

    if (tile_size >= UNROLL_SIZE)
    {
        half8 r00 = data_in[0];
        half8 r01 = data_in[1];
        half8 r02 = data_in[2];
        half8 r03 = data_in[3];
        half8 r04 = data_in[4];
        half8 r05 = data_in[5];
        half8 r06 = data_in[6];
        half8 r07 = data_in[7];
        data_in += UNROLL_SIZE;

        half8 q00 = __builtin_shave_cmu_min_f16_rr_half8(r00, zeros) * x_vec;
        half8 q01 = __builtin_shave_cmu_min_f16_rr_half8(r01, zeros);
        half8 q02 = __builtin_shave_cmu_min_f16_rr_half8(r02, zeros);
        half8 q03 = __builtin_shave_cmu_min_f16_rr_half8(r03, zeros);
        half8 q04 = __builtin_shave_cmu_min_f16_rr_half8(r04, zeros);
        half8 q05 = __builtin_shave_cmu_min_f16_rr_half8(r05, zeros);
        half8 q06 = __builtin_shave_cmu_min_f16_rr_half8(r06, zeros);

        for (i = UNROLL_SIZE; i < tile_size - (UNROLL_SIZE - 1); i += UNROLL_SIZE)
        {
            half8 r10 = data_in[0];
            half8 r11 = data_in[1];
            half8 r12 = data_in[2];
            half8 r13 = data_in[3];
            half8 r14 = data_in[4];
            half8 r15 = data_in[5];
            half8 r16 = data_in[6];
            half8 r17 = data_in[7];
            data_in += UNROLL_SIZE;

            half8 q10 = __builtin_shave_cmu_min_f16_rr_half8(r10, zeros) * x_vec;
            half8 q11 = __builtin_shave_cmu_min_f16_rr_half8(r11, zeros);
            half8 q12 = __builtin_shave_cmu_min_f16_rr_half8(r12, zeros);
            half8 q13 = __builtin_shave_cmu_min_f16_rr_half8(r13, zeros);
            half8 q14 = __builtin_shave_cmu_min_f16_rr_half8(r14, zeros);
            half8 q15 = __builtin_shave_cmu_min_f16_rr_half8(r15, zeros);
            half8 q16 = __builtin_shave_cmu_min_f16_rr_half8(r16, zeros);

            data_out[0] = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + q00;
            data_out[1] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + q01 * x_vec;
            data_out[2] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + q02 * x_vec;
            data_out[3] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + q03 * x_vec;
            data_out[4] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + q04 * x_vec;
            data_out[5] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + q05 * x_vec;
            data_out[6] = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + q06 * x_vec;
            data_out[7] = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(r07, zeros);
            data_out += UNROLL_SIZE;

            r00 = r10; q00 = q10;
            r01 = r11; q01 = q11;
            r02 = r12; q02 = q12;
            r03 = r13; q03 = q13;
            r04 = r14; q04 = q14;
            r05 = r15; q05 = q15;
            r06 = r16; q06 = q16;
            r07 = r17;
        }

        data_out[0] = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + q00;
        data_out[1] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + q01 * x_vec;
        data_out[2] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + q02 * x_vec;
        data_out[3] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + q03 * x_vec;
        data_out[4] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + q04 * x_vec;
        data_out[5] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + q05 * x_vec;
        data_out[6] = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + q06 * x_vec;
        data_out[7] = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(r07, zeros);
        data_out += UNROLL_SIZE;
    }

    for (; i < tile_size; ++i)
    {
        half8 r = *data_in++;
        *data_out++ = __builtin_shave_cmu_max_f16_rr_half8(r, zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(r, zeros);
    }

#endif // MA2480

}

void HWC_prelu_fp16(half8 * __restrict__ data_in,
                    half8 * __restrict__ data_out,
                    half8 * __restrict__ weights,
                    half8 * __restrict__ bias,
                    s32 no_lines, s32 line_size, half x,
                    void *parameters)
{
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);

    const half8 zeros = (half8)0.0;

        for (s32 line_i = 0; line_i < line_size; ++line_i)
        {
            half8 w = weights[line_i];

            half8 * __restrict__ in = &data_in[line_i];
            half8 * __restrict__ out = &data_out[line_i];

            s32 i = 0;
            if (no_lines >= UNROLL_SIZE)
            {
                half8 r00 = in[0 * line_size];
                half8 r01 = in[1 * line_size];
                half8 r02 = in[2 * line_size];
                half8 r03 = in[3 * line_size];
                half8 r04 = in[4 * line_size];
                half8 r05 = in[5 * line_size];
                half8 r06 = in[6 * line_size];
                half8 r07 = in[7 * line_size];
                in += UNROLL_SIZE * line_size;

                half8 q00 = __builtin_shave_cmu_min_f16_rr_half8(r00, zeros);
                half8 q01 = __builtin_shave_cmu_min_f16_rr_half8(r01, zeros);
                half8 q02 = __builtin_shave_cmu_min_f16_rr_half8(r02, zeros);
                half8 q03 = __builtin_shave_cmu_min_f16_rr_half8(r03, zeros);

                for (i = UNROLL_SIZE; i < no_lines - (UNROLL_SIZE - 1); i += UNROLL_SIZE)
                {
                    half8 r10 = in[0 * line_size];
                    half8 r11 = in[1 * line_size];
                    half8 r12 = in[2 * line_size];
                    half8 r13 = in[3 * line_size];
                    half8 r14 = in[4 * line_size];
                    half8 r15 = in[5 * line_size];
                    half8 r16 = in[6 * line_size];
                    half8 r17 = in[7 * line_size];
                    in += UNROLL_SIZE * line_size;

                    half8 q10 = __builtin_shave_cmu_min_f16_rr_half8(r10, zeros);
                    half8 q11 = __builtin_shave_cmu_min_f16_rr_half8(r11, zeros);
                    half8 q12 = __builtin_shave_cmu_min_f16_rr_half8(r12, zeros);
                    half8 q13 = __builtin_shave_cmu_min_f16_rr_half8(r13, zeros);

                    out[0 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + q00 * w;
                    out[1 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + q01 * w;
                    out[2 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + q02 * w;
                    out[3 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + q03 * w;
                    out[4 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r04, zeros);
                    out[5 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r05, zeros);
                    out[6 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r06, zeros);
                    out[7 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r07, zeros);
                    out += UNROLL_SIZE * line_size;

                    r00 = r10; q00 = q10;
                    r01 = r11; q01 = q11;
                    r02 = r12; q02 = q12;
                    r03 = r13; q03 = q13;
                    r04 = r14;
                    r05 = r15;
                    r06 = r16;
                    r07 = r17;
                }

                out[0 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + q00 * w;
                out[1 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + q01 * w;
                out[2 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + q02 * w;
                out[3 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + q03 * w;
                out[4 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r04, zeros);
                out[5 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r05, zeros);
                out[6 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r06, zeros);
                out[7 * line_size] = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r07, zeros);
                out += UNROLL_SIZE * line_size;
            }

            for (; i < no_lines; ++i)
            {
                half8 r = in[0];
                out[0] = __builtin_shave_cmu_max_f16_rr_half8(r, zeros) + w * __builtin_shave_cmu_min_f16_rr_half8(r, zeros);
                in += line_size;
                out += line_size;
            }
        }

}

void tanh_fp16(half8 * __restrict__ data_in,
               half8 * __restrict__ data_out,
               half8 * __restrict__ weights,
               half8 * __restrict__ bias,
               s32 no_lines, s32 line_size, half x,
               void *parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);
    // Clamp the input to avoid fp16 precision overflow when computing exp.
    // This should not affect the results
    half8 upper_bound =   5.5f;
    half8 lower_bound = -10.5f;


    // Compute tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // = (2^(2x/ln(2)) - 1) / (2^(2x/ln(2)) + 1)
    const unsigned short inv_ln2_mul_2 = 0x41c5;
    const half inv_ln2_mul_2_h = *reinterpret_cast<const half *>(&inv_ln2_mul_2);

#pragma clang loop unroll_count(4)
    for(s32 i = 0; i < (no_lines * line_size); ++i)
    {
        data_out[i] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i], lower_bound, upper_bound);
        data_out[i] = data_out[i] * inv_ln2_mul_2_h;
        exp2_vec(data_out[i], data_out[i]);
    }

#pragma clang loop unroll_count(4)
    for(s32 i = 0; i < (no_lines * line_size); ++i)
    {
        data_out[i] = (data_out[i] - 1.0f) / (data_out[i] + 1.0f);
    }
}

void sigmoid_fp16(half8 * __restrict__ data_in,
                  half8 * __restrict__ data_out,
                  half8 * __restrict__ weights,
                  half8 * __restrict__ bias,
                  s32 no_lines, s32 line_size, half x,
                  void *parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);

    // Compute sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x/ln(2)))
    const unsigned short negative_inv_ln2 = 0xbdc6;
    const half negative_inv_ln2_h = *reinterpret_cast<const half *>(&negative_inv_ln2);

#pragma clang loop unroll_count(8)
    for(s32 i = 0; i < (no_lines * line_size); ++i)
    {
        data_out[i] = data_in[i] * negative_inv_ln2_h;
        exp2_vec(data_out[i], data_out[i]);
    }

#pragma clang loop unroll_count(8)
    for(s32 i = 0; i < (no_lines * line_size); ++i)
    {
        data_out[i] = (1.0f / (1.0f + (data_out[i])));
    }
}

void HWC_bias_fp16(half8 * __restrict__ data_in,
                    half8 * __restrict__ data_out,
                    half8 * __restrict__ weights,
                    half8 * __restrict__ bias,
                    s32 no_lines, s32 line_size, half x,
                    void *parameters)
{
    UNUSED(weights);
    UNUSED(x);
    UNUSED(parameters);

    half8 * __restrict__ r_in;
    half8 * __restrict__ r_out;

    int bias_i = 0;
    for(; bias_i < line_size - 3; bias_i += 4)
    {
        int i = 0;
        r_in = data_in + bias_i;
        r_out = data_out + bias_i;
        half8 r_b0 = bias[bias_i];
        half8 r_b1 = bias[bias_i+1];
        half8 r_b2 = bias[bias_i+2];
        half8 r_b3 = bias[bias_i+3];

        if(no_lines > 2)
        {
            half8 rr00 = r_in[0] + r_b0;
            half8 rr01 = r_in[1] + r_b1;
            half8 rr02 = r_in[2] + r_b2;
            half8 rr03 = r_in[3] + r_b3;
            r_in += line_size;
            half8 rr10 = r_in[0];
            half8 rr11 = r_in[1];
            half8 rr12 = r_in[2] + r_b2;
            half8 rr13 = r_in[3] + r_b3;
            r_in += line_size;

#pragma clang loop unroll_count(4)
            for(; i < no_lines-2; i ++, r_in += line_size, r_out += line_size)
            {
                half8 rr20 = r_in[0];
                half8 rr21 = r_in[1];
                half8 rr22 = r_in[2];
                half8 rr23 = r_in[3];
                r_out[0] = rr00;
                r_out[1] = rr01;
                r_out[2] = rr02;
                r_out[3] = rr03;
                rr00 = rr10+r_b0;
                rr01 = rr11+r_b1;
                rr02 = rr12;
                rr03 = rr13;

                rr10 = rr20;
                rr11 = rr21;
                rr12 = rr22+r_b2;
                rr13 = rr23+r_b3;
            }
            r_in -= 2 * line_size;
        }
        for(; i < no_lines; ++i, r_in += line_size, r_out += line_size)
        {
            r_out[0] = r_in[0] + r_b0;
            r_out[1] = r_in[1] + r_b1;
            r_out[2] = r_in[2] + r_b2;
            r_out[3] = r_in[3] + r_b3;
        }
    }

    for(; bias_i < line_size; bias_i ++)
    {
        int i = 0;
        r_in = data_in + bias_i;
        r_out = data_out + bias_i;
        half8 r_b0 = bias[bias_i];

        if(no_lines > 2)
        {
            half8 rr0 = r_in[0] + r_b0;
            r_in += line_size;
            half8 rr1 = r_in[0];
            r_in += line_size;

#pragma clang loop unroll_count(8)
            for(; i < no_lines-2; i ++)
            {
                half8 rr2 = r_in[0];
                r_in += line_size;
                r_out[0] = rr0;
                r_out += line_size;
                rr0 = rr1+r_b0;

                rr1 = rr2;
            }
            r_in -= 2 * line_size;
        }

        for(; i < no_lines; ++i)
        {
            r_out[0] = r_in[0] + r_b0;
            r_in  += line_size;
            r_out += line_size;
        }
    }
}

void HWC_scale_fp16(half8 * __restrict__ data_in,
                    half8 * __restrict__ data_out,
                    half8 * __restrict__ weights,
                    half8 * __restrict__ bias,
                    s32 no_lines, s32 line_size, half x,
                    void *parameters)
{
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);

    half8 * __restrict__ r_in;
    half8 * __restrict__ r_out;

    int bias_i = 0;
    for(; bias_i < line_size - 3; bias_i += 4)
    {
        int i = 0;
        r_in = data_in + bias_i;
        r_out = data_out + bias_i;
        half8 r_w0 = weights[bias_i];
        half8 r_w1 = weights[bias_i+1];
        half8 r_w2 = weights[bias_i+2];
        half8 r_w3 = weights[bias_i+3];

        if(no_lines > 2)
        {
            half8 rr00 = r_in[0] * r_w0;
            half8 rr01 = r_in[1] * r_w1;
            half8 rr02 = r_in[2] * r_w2;
            half8 rr03 = r_in[3] * r_w3;
            r_in += line_size;
            half8 rr10 = r_in[0];
            half8 rr11 = r_in[1];
            half8 rr12 = r_in[2] * r_w2;
            half8 rr13 = r_in[3] * r_w3;
            r_in += line_size;

#pragma clang loop unroll_count(4)
            for(; i < no_lines-2; i ++, r_in += line_size, r_out += line_size)
            {
                half8 rr20 = r_in[0];
                half8 rr21 = r_in[1];
                half8 rr22 = r_in[2];
                half8 rr23 = r_in[3];
                r_out[0] = rr00;
                r_out[1] = rr01;
                r_out[2] = rr02;
                r_out[3] = rr03;
                rr00 = rr10*r_w0;
                rr01 = rr11*r_w1;
                rr02 = rr12;
                rr03 = rr13;

                rr10 = rr20;
                rr11 = rr21;
                rr12 = rr22*r_w2;
                rr13 = rr23*r_w3;
            }
            r_in -= 2 * line_size;
        }
        for(; i < no_lines; ++i, r_in += line_size, r_out += line_size)
        {
            r_out[0] = r_in[0] * r_w0;
            r_out[1] = r_in[1] * r_w1;
            r_out[2] = r_in[2] * r_w2;
            r_out[3] = r_in[3] * r_w3;
        }
    }

    for(; bias_i < line_size; bias_i ++)
    {
        int i = 0;
        r_in = data_in + bias_i;
        r_out = data_out + bias_i;
        half8 r_w0 = weights[bias_i];

        if(no_lines > 2)
        {
            half8 rr0 = r_in[0] * r_w0;
            r_in += line_size;
            half8 rr1 = r_in[0];
            r_in += line_size;

#pragma clang loop unroll_count(8)
            for(; i < no_lines-2; i ++)
            {
                half8 rr2 = r_in[0];
                r_in += line_size;
                r_out[0] = rr0;
                r_out += line_size;
                rr0 = rr1*r_w0;

                rr1 = rr2;
            }
            r_in -= 2 * line_size;
        }

        for(; i < no_lines; ++i)
        {
            r_out[0] = r_in[0] * r_w0;
            r_in  += line_size;
            r_out += line_size;
        }
    }
}

void HWC_scaleShift_fp16(half8 * __restrict__ data_in,
                            half8 * __restrict__ data_out,
                            half8 * __restrict__ weights,
                            half8 * __restrict__ bias,
                            s32 no_lines, s32 line_size, half x,
                            void *parameters)
{
    UNUSED(x);
    UNUSED(parameters);
    half8 * __restrict__ r_in;
    half8 * __restrict__ r_out;

    int bias_i = 0;
    for(; bias_i < line_size - 3; bias_i += 4)
    {
        int i = 0;
        r_in = data_in + bias_i;
        r_out = data_out + bias_i;
        half8 r_w0 = weights[bias_i];
        half8 r_w1 = weights[bias_i+1];
        half8 r_w2 = weights[bias_i+2];
        half8 r_w3 = weights[bias_i+3];

        half8 r_b0 = bias[bias_i];
        half8 r_b1 = bias[bias_i+1];
        half8 r_b2 = bias[bias_i+2];
        half8 r_b3 = bias[bias_i+3];

        if(no_lines > 2)
        {
            half8 rr00 = r_in[0] * r_w0 + r_b0;
            half8 rr01 = r_in[1] * r_w1 + r_b1;
            half8 rr02 = r_in[2] * r_w2 + r_b2;
            half8 rr03 = r_in[3] * r_w3 + r_b3;
            r_in += line_size;
            half8 rr10 = r_in[0];
            half8 rr11 = r_in[1];
            half8 rr12 = r_in[2] * r_w2 + r_b2;
            half8 rr13 = r_in[3] * r_w3 + r_b3;
            r_in += line_size;

#pragma clang loop unroll_count(3)
            for(; i < no_lines-2; i ++, r_in += line_size, r_out += line_size)
            {
                half8 rr20 = r_in[0];
                half8 rr21 = r_in[1];
                half8 rr22 = r_in[2];
                half8 rr23 = r_in[3];
                r_out[0] = rr00;
                r_out[1] = rr01;
                r_out[2] = rr02;
                r_out[3] = rr03;
                rr00 = rr10*r_w0 + r_b0;
                rr01 = rr11*r_w1 + r_b1;
                rr02 = rr12;
                rr03 = rr13;

                rr10 = rr20;
                rr11 = rr21;
                rr12 = rr22*r_w2 + r_b2;
                rr13 = rr23*r_w3 + r_b3;
            }
            r_in -= 2 * line_size;
        }
        for(; i < no_lines; ++i, r_in += line_size, r_out += line_size)
        {
            r_out[0] = r_in[0] * r_w0 + r_b0;
            r_out[1] = r_in[1] * r_w1 + r_b1;
            r_out[2] = r_in[2] * r_w2 + r_b2;
            r_out[3] = r_in[3] * r_w3 + r_b3;
        }
    }

    for(; bias_i < line_size; bias_i ++)
    {
        int i = 0;
        r_in = data_in + bias_i;
        r_out = data_out + bias_i;
        half8 r_w0 = weights[bias_i];
        half8 r_b0 = bias[bias_i];

        if(no_lines > 3)
        {
            half8 rr0 = r_in[0] * r_w0 + r_b0;
            r_in += line_size;
            half8 rr1 = r_in[0] * r_w0;
            r_in += line_size;
            half8 rr2 = r_in[0];
            r_in += line_size;

#pragma clang loop unroll_count(4)
            for(; i < no_lines-3; i += 1)
            {
                half8 rr3 = r_in[0];
                r_in += line_size;
                r_out[0] = rr0;
                r_out += line_size;

                rr0 = rr1 + r_b0;
                rr1 = rr2 * r_w0;
                rr2 = rr3;
            }
            r_in -= 3 * line_size;
        }

        for(; i < no_lines; ++i)
        {
            r_out[0] = r_in[0] * r_w0 + r_b0;
            r_in  += line_size;
            r_out += line_size;
        }
    }
}

void clamp_fp16(half8 * __restrict__ data_in,
                half8 * __restrict__ data_out,
                half8 * __restrict__ weights,
                half8 * __restrict__ bias,
                s32 no_lines, s32 line_size, half x,
                void *parameters
               )
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);

    const half8 min_val = (half8)reinterpret_cast<t_ClampLayerParams *>(parameters)->min;
    const half8 max_val = (half8)reinterpret_cast<t_ClampLayerParams *>(parameters)->max;

    s32 i = 0;
    for(i = 0; i < (((no_lines * line_size) / UNROLL_SIZE) * UNROLL_SIZE); i += UNROLL_SIZE)
    {
        data_out[i + 0] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 0], min_val, max_val);
        data_out[i + 1] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 1], min_val, max_val);
        data_out[i + 2] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 2], min_val, max_val);
        data_out[i + 3] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 3], min_val, max_val);
        data_out[i + 4] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 4], min_val, max_val);
        data_out[i + 5] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 5], min_val, max_val);
        data_out[i + 6] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 6], min_val, max_val);
        data_out[i + 7] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i + 7], min_val, max_val);
    }

    for(; i < (no_lines * line_size); ++i)
        data_out[i] = __builtin_shave_cmu_clampab_f16_rrr_half8(data_in[i], min_val, max_val);
}

void eluFp16(half8 * __restrict__ data_in,
             half8 * __restrict__ data_out,
             half8 * __restrict__ weights,
             half8 * __restrict__ bias,
             s32 no_lines, s32 line_size, half x,
             void *parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(parameters);

    // Compute elu(x) = x for                    x >  0
    //                = alpha * (exp(x) - 1) for x <= 0
    // using exp(x) = 2^(x/ln(2))

    const half alpha = x;
    const half8 one  = (half8)1.0f;
    const half8 zero = (half8)0.0f;

    const unsigned short inv_ln2 = 0x3dc6;
    const half inv_ln2_h = *reinterpret_cast<const half *>(&inv_ln2);
    const half8 vinv_ln2 = (half8)inv_ln2_h;

#pragma clang loop unroll_count(8)
    for(s32 i = 0; i < (no_lines * line_size); ++i)
    {
        half8 min = __builtin_shave_cmu_min_f16_rr_half8(data_in[i], zero);
        half8 max = __builtin_shave_cmu_max_f16_rr_half8(data_in[i], zero);

        half8 exp_x = min * vinv_ln2;
        exp2_vec(exp_x, exp_x);

        data_out[i] = max + alpha * (exp_x - one);
    }
}

void relu_fp16(half8 * __restrict__ data_in,
               half8 * __restrict__ data_out,
               half8 * __restrict__ weights,
               half8 * __restrict__ bias,
               s32 no_lines, s32 line_size, half x,
               void *parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(parameters);

    const half8  x_vec = (x <= half(0.0)) ? (half8)(X_MAX) : (half8)x;

    s32 i = 0;
    for(i = 0; i < (((no_lines * line_size) / UNROLL_SIZE) * UNROLL_SIZE); i += UNROLL_SIZE)
    {
        data_out[i + 0] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 0], x_vec);
        data_out[i + 1] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 1], x_vec);
        data_out[i + 2] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 2], x_vec);
        data_out[i + 3] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 3], x_vec);
        data_out[i + 4] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 4], x_vec);
        data_out[i + 5] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 5], x_vec);
        data_out[i + 6] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 6], x_vec);
        data_out[i + 7] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 7], x_vec);
    }

    for(; i < (no_lines * line_size); ++i)
        data_out[i] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i], x_vec);
}

void power_fp16(half8 * __restrict__ data_in,
                half8 * __restrict__ data_out,
                half8 * __restrict__ weights,
                half8 * __restrict__ bias,
                s32 no_lines,
                s32 line_size,
                half x,
                void *parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(x);

    const half8 shift = (half8)reinterpret_cast<t_PowerLayerParams *>(parameters)->shift;
    const float shift_scal = reinterpret_cast<t_PowerLayerParams *>(parameters)->shift;

    const half8 scale = (half8)reinterpret_cast<t_PowerLayerParams *>(parameters)->scale;
    const float scale_scal = reinterpret_cast<t_PowerLayerParams *>(parameters)->scale;

    const half8 power = (half8)reinterpret_cast<t_PowerLayerParams *>(parameters)->power;
    const float power_scal = reinterpret_cast<t_PowerLayerParams *>(parameters)->power;

    // Compute power(x) = (shift + scale * x)^power
    int num_elements = no_lines * line_size;

    if ((scale_scal == 1.0f) && (power_scal == 1.0f) && (shift_scal == 0.0f)) // out = in (copying)
    {
        // Do not do anything; src is already copied to dst CMX tile
        return;
    }

    if(power_scal == 0.0f) // power == 0
    {
        half8 vfill = (half8)1.0f;
#pragma clang loop unroll_count(8)
        for(s32 i = 0; i < (num_elements); ++i)
        {
            data_out[i] = vfill;
        }
    }
    else
    {
        bool is_power_integer = floorf(fabs(power_scal)) == fabs(power_scal);
        const s32 integer_power = fabs(power_scal);

        if (is_power_integer) // power is integer
        {
            if (integer_power == 1) // power == 1
            {
                if ((scale_scal == -1.0f) && (shift_scal == 0.0f)) // out = -in
                {
                    half8 vec;
                    if (num_elements)
                    {
                        vec = -data_in[0];
                    }
#pragma clang loop unroll_count(4)
                    for (int i = 1; i < (num_elements); i++)
                    {
                        data_out[i-1] = vec;
                        vec = -data_in[i];
                    }
                    if (num_elements)
                    {
                        data_out[num_elements-1] = vec;
                    }
                }
                else
                {
                    half8 v;

                    if (num_elements)
                    {
                        v = scale * data_in[0];
                    }
#pragma clang loop unroll_count(8)
                    for (s32 i = 1; i < num_elements; i++)
                    {
                        half8 vres = v + shift;
                        v = scale * data_in[i];
                        data_out[i-1] = vres;
                    }
                    if (num_elements)
                    {
                        data_out[num_elements-1] = v + shift;
                    }
                }

            }
            else if (integer_power == 2) // power == 2
            {
                half8 base;

                if (num_elements)
                {
                    base = shift + scale * data_in[0];
                }
#pragma clang loop unroll_count(8)
                for (s32 i = 1; i < (num_elements); i++)
                {
                    half8 vres = (base) * (base);
                    base = shift + scale * data_in[i];
                    data_out[i-1] = vres;
                }
                if (num_elements)
                {
                    data_out[num_elements-1] = (base)*(base);
                }
            }
            else if (integer_power == 3) // power == 3
            {
                half8 vin;
                half8 base;
                half8 vres;

                // 0 iteration
                if (num_elements)
                {
                    half8 vin = data_in[0];
                    base = shift + scale * vin;
                    vres = base * base * base;
                }
                // 1 iteration
                if (num_elements > 1)
                {
                    vin = data_in[1];
                    base = shift + scale * vin;
                }
#pragma clang loop unroll_count(8)
                for (s32 i = 2; i < (num_elements); i++)
                {
                    data_out[i-2] = vres;
                    vres = base * base * base;
                    half8 vin = data_in[i];
                    base = shift + scale * vin;
                }
                // (num_elements-2) iteration
                if (num_elements > 1)
                {
                    data_out[num_elements-2] = vres;
                }
                // (num_elements-1) iteration
                if (num_elements > 0)
                {
                    data_out[num_elements-1] = (base)*(base)*(base);
                }
            }
            else // general integer power
            {
                s32 i = 0;
#pragma clang loop unroll_count(1)
                for (; i < num_elements-7; i+=8)
                {
                    half8 base0 = shift + scale * data_in[i];
                    half8 base1 = shift + scale * data_in[i+1];
                    half8 base2 = shift + scale * data_in[i+2];
                    half8 base3 = shift + scale * data_in[i+3];
                    half8 base4 = shift + scale * data_in[i+4];
                    half8 base5 = shift + scale * data_in[i+5];
                    half8 base6 = shift + scale * data_in[i+6];
                    half8 base7 = shift + scale * data_in[i+7];

                    half8 res0 = base0;
                    half8 res1 = base1;
                    half8 res2 = base2;
                    half8 res3 = base3;
                    half8 res4 = base4;
                    half8 res5 = base5;
                    half8 res6 = base6;
                    half8 res7 = base7;

                    int p = 1;
                    for (; (p << 1) <= integer_power; p<<=1)
                    {
                        res0 = res0 * res0;
                        res1 = res1 * res1;
                        res2 = res2 * res2;
                        res3 = res3 * res3;
                        res4 = res4 * res4;
                        res5 = res5 * res5;
                        res6 = res6 * res6;
                        res7 = res7 * res7;
                    }

                    for (; p < integer_power; p++)
                    {
                        res0 = res0 * base0;
                        res1 = res1 * base1;
                        res2 = res2 * base2;
                        res3 = res3 * base3;
                        res4 = res4 * base4;
                        res5 = res5 * base5;
                        res6 = res6 * base6;
                        res7 = res7 * base7;
                    }
                    data_out[i+0] = res0;
                    data_out[i+1] = res1;
                    data_out[i+2] = res2;
                    data_out[i+3] = res3;
                    data_out[i+4] = res4;
                    data_out[i+5] = res5;
                    data_out[i+6] = res6;
                    data_out[i+7] = res7;
                }

                for (; i < num_elements; i++)
                {
                    half8 base = shift + scale * data_in[i];

                    half8 res = base;
                    for (int p = 0; p < integer_power-1; p++)
                    {
                        res = res * base;
                    }
                    data_out[i] = res;
                }
            }

            if (power_scal < 0.0f)
            {
                half8 v;
                if (num_elements)
                {
                    v = data_in[0];
                }
#pragma clang loop unroll_count(8)
                for (s32 i = 1; i < (num_elements); i++)
                {
                    data_out[i-1] = 1.f / v;
                    v = data_in[i];
                }
                if (num_elements)
                {
                    data_out[num_elements-1] = 1.f / v;
                }
            }
        }
        else // general case
        {
            half8 base_log = (half8)0, base_mult = (half8)0, base = (half8)0;

            // 0 iteration
            if (num_elements > 0)
            {
                base = shift + scale * data_in[0];
                log2_vec(base, base_log);
            }
            // 1 iteration
            if (num_elements > 1)
            {
                base = shift + scale * data_in[1];
            }

#pragma clang loop unroll_count(8)
            for (s32 i = 2; i < (num_elements); ++i)
            {
                base_mult = base_log * power;// 2 stage
                exp2_vec(base_mult, data_out[i-2]);

                log2_vec(base, base_log);// 1 stage

                base = shift + scale * data_in[i];// 0 stage
            }

            // (num_elements-2) iteration
            if (num_elements > 1)
            {
                base_mult = base_log * power;
                exp2_vec(base_mult, data_out[num_elements-2]);
            }
            // (num_elements-1) iteration
            if (num_elements > 0)
            {
                log2_vec(base, base_log);
                base_mult = base_log * power;
                exp2_vec(base_mult, data_out[num_elements-1]);
            }
        }
    }

}

// Swish(x) = x / (1 + exp(-beta * x))
static void swish_fp16(const half8 * __restrict__ data_in,
                             half8 * __restrict__ data_out,
                       const half8 * __restrict__ weights,
                       const half8 * __restrict__ bias,
                       const s32     no_lines,
                       const s32     line_size,
                       const half    coefficient,
                       const void  * parameters)
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(coefficient);

    using nn::shave_lib::t_SwishLayerParams;
    const t_SwishLayerParams* swishParams = reinterpret_cast<const t_SwishLayerParams*>(parameters);
    const half beta = static_cast<half>(swishParams->beta);

    const uint16_t inv_ln2 = 0x3dc5;
    const half inv_ln2_h = *(const half*)& inv_ln2;

    const half8 one = (half8) 1.0h;
    const half8 nbeta_inv_ln2 = (half8)(-beta * inv_ln2_h);

    const s32 tile_size = no_lines * line_size;

#pragma clang loop unroll_count(16)
    for (s32 i = 0; i < tile_size; ++i)
    {
        const half8 x = data_in[i];

        half8 exponent = __builtin_shave_vau_mul_f16_rr(nbeta_inv_ln2, x);
        exp2_vec(exponent, exponent);

        half8 denom = __builtin_shave_vau_add_f16_rr(exponent, one);
        half8 inv_denom = 1 / denom;

        half8 res = __builtin_shave_vau_mul_f16_rr(x, inv_denom);

        data_out[i] = res;
    }
}

void hswish_fp16(half8 * __restrict__ data_in,
                 half8 * __restrict__ data_out,
                 half8 * __restrict__ weights,
                 half8 * __restrict__ bias,
                 s32 no_lines, s32 line_size, half x,
                 void *parameters
                )
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);

    const half8 add_val_3 = 3.0f;
    const half8 max_val_6 = 6.0f;

    s32 tile_size = line_size * no_lines;
    s32 i = 0;
    for(; i < ((tile_size / UNROLL_SIZE) * UNROLL_SIZE); i += UNROLL_SIZE)
    {
        data_out[i + 0] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 0] + add_val_3, max_val_6) * data_in[i + 0] / max_val_6;
        data_out[i + 1] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 1] + add_val_3, max_val_6) * data_in[i + 1] / max_val_6;
        data_out[i + 2] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 2] + add_val_3, max_val_6) * data_in[i + 2] / max_val_6;
        data_out[i + 3] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 3] + add_val_3, max_val_6) * data_in[i + 3] / max_val_6;
        data_out[i + 4] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 4] + add_val_3, max_val_6) * data_in[i + 4] / max_val_6;
        data_out[i + 5] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 5] + add_val_3, max_val_6) * data_in[i + 5] / max_val_6;
        data_out[i + 6] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 6] + add_val_3, max_val_6) * data_in[i + 6] / max_val_6;
        data_out[i + 7] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i + 7] + add_val_3, max_val_6) * data_in[i + 7] / max_val_6;
    }

    for(; i < tile_size; ++i)
        data_out[i] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[i] + add_val_3, max_val_6) * data_in[i] / max_val_6;
}

static void softplus_fp16(const half8* __restrict__ data_in,
                          half8*       __restrict__ data_out,
                          const half8* __restrict__ weights,
                          const half8* __restrict__ biases,
                          s32 no_lines, s32 line_size, half x,
                          void *parameters)
{
    UNUSED(weights);
    UNUSED(biases);
    UNUSED(x);
    UNUSED(parameters);

// SoftPlus(x) = log(1 + exp(x))
//             = log1p(exp(x)) | x << 0
//             = x + log1p(exp(-x)) | x >> 0

    const int tile_size = line_size * no_lines;

    const float8 one = (float8)1.0f;

    int i = 0;
    if (tile_size >= 2)
    {
        float8 x0 = mvuConvert_float8(data_in[i + 0]);
        float8 x1 = mvuConvert_float8(data_in[i + 1]);
        float8 tmp0 = expf8(x0);
        float8 tmp1 = expf8(x1);
        for (i = 2; i < tile_size - 1; i += 2)
        {
            float8 x2 = mvuConvert_float8(data_in[i + 0]);
            float8 x3 = mvuConvert_float8(data_in[i + 1]);
            float8 tmp2 = expf8(x2);
            float8 tmp3 = expf8(x3);
            float8 res0 = logf8(one + tmp0);
            float8 res1 = logf8(one + tmp1);
            data_out[i - 2] = mvuConvert_half8(res0);
            data_out[i - 1] = mvuConvert_half8(res1);
            tmp0 = tmp2;
            tmp1 = tmp3;
        }
        float8 res0 = logf8(one + tmp0);
        float8 res1 = logf8(one + tmp1);
        data_out[i - 2] = mvuConvert_half8(res0);
        data_out[i - 1] = mvuConvert_half8(res1);
    }
    for (; i < tile_size; ++i)
    {
        float8 x = mvuConvert_float8(data_in[i]);
        float8 res = logf8(one + expf8(x));
        data_out[i] = mvuConvert_half8(res);
    }
}

void mish_fp16(half8 * __restrict__ data_in,
                 half8 * __restrict__ data_out,
                 half8 * __restrict__ weights,
                 half8 * __restrict__ bias,
                 s32 no_lines, s32 line_size, half x,
                 void *parameters
                )
{
    UNUSED(weights);
    UNUSED(bias);
    UNUSED(x);
    UNUSED(parameters);

    // Mish(x) = x * tanh(ln(1 + e^x))

    const uint16_t inv_ln2 = 0x3dc5;
    const half inv_ln2_h = *(const half*)&inv_ln2;
    const unsigned short ln2 = 0x398c;
    const half ln2_h = *reinterpret_cast<const half *>(&ln2);

    // Compute tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // = (2^(2x/ln(2)) - 1) / (2^(2x/ln(2)) + 1)
    const uint16_t inv_ln2_mul_2 = 0x41c5;
    const half inv_ln2_mul_2_h = *(const half*)&inv_ln2_mul_2;
    const half one = (half)1.0f;
    const half8 upper_bound =   5.5f;
    const half8 lower_bound = -10.5f;

    const int numVectors = no_lines * line_size;
    if (numVectors == 0)
        return;

    half8 tmp = data_in[0];

    // 1 + e^x
    half8 exp_x = tmp * inv_ln2_h;
    exp2_vec(exp_x, exp_x);
    tmp = exp_x + 1.0f;

    // ln(1 + e^x)
    log2_vec(tmp, tmp);
    tmp = tmp * ln2_h;

    // tanh(ln(1 + e^x))
    // Clamp the input to avoid fp16 precision overflow when computing exp.
    // This should not affect the results
    tmp = __builtin_shave_cmu_clampab_f16_rrr_half8(tmp, lower_bound, upper_bound);
    tmp = tmp * inv_ln2_mul_2_h;
    exp2_vec(tmp, tmp);
    tmp = (tmp - one) / (tmp + one);

    for(s32 i = 1; i < numVectors; ++i)
    {
        // x[i-1] * tanh(ln(1 + e^x[i-1]))
        data_out[i - 1] = data_in[i - 1] * tmp;

        tmp = data_in[i];

        // 1 + e^x
        half8 exp_x = tmp * inv_ln2_h;
        exp2_vec(exp_x, exp_x);
        tmp = exp_x + 1.0f;

        // ln(1 + e^x)
        log2_vec(tmp, tmp);
        tmp = tmp * ln2_h;

        // tanh(ln(1 + e^x))
        tmp = __builtin_shave_cmu_clampab_f16_rrr_half8(tmp, lower_bound, upper_bound);
        tmp = tmp * inv_ln2_mul_2_h;
        exp2_vec(tmp, tmp);
        tmp = (tmp - one) / (tmp + one);
    }

    data_out[numVectors - 1] = data_in[numVectors - 1] * tmp;
}

bool getHWCOperation(const t_PostOps postOpType, operation_hwc_type &operation, half &x, void *pparams);
bool getHCWOperation(const t_PostOps postOpType, operation_hcw_type &operation, half &x, void *pparams);

void hwc_postOps_3D_core(t_HWCPostOps3DParams *params)
{
    DmaAlShave dmaTask;

    const half *input   = params->input;
    half *output        = params->output;
    const half *weights = params->weights;

    const half *bias    = params->bias;
    u32  width          = params->width;
    u32  height         = params->height;
    u32  in_stride      = params->in_step;
    u32  out_stride     = params->out_step;
    half x              = half(0);
    void *pparams       = params->params;

    const auto CMX_SIZE = params->availableCmxBytes;

    u32 in_stride_in_bytes  = in_stride * sizeof(half);
    u32 out_stride_in_bytes = out_stride * sizeof(half);

    operation_hwc_type operation = nullptr;

    bool success = getHWCOperation(params->postOpType, operation, x, pparams);
    if (!success) return;

    s32 cmx_bl8h    = (CMX_SIZE >> 3) / sizeof(half); // Size of cmx in blocks of 8 halfs.
    s32 w_bl8h      = (width + 7) / 8; // Size of width in blocks of 8 halfs.

    s32 no_chunks = ((!!bias + !!weights + 1) * w_bl8h + cmx_bl8h - 1) / cmx_bl8h;

    int chunk_size_bl8h = (w_bl8h + no_chunks - 1) / no_chunks;
    int chunk_size_bytes = chunk_size_bl8h * sizeof(half8);

    half8 *v_bias    = (half8 *)(params->cmxData);
    half8 *v_weights = (half8 *)(params->cmxData);
    half8 *v_in_out0 = (half8 *)(params->cmxData);

    int left_cmx = CMX_SIZE;

    if ((bias != 0) && (weights != 0))
    {
        // Make space for weights and bias
        v_bias    = (half8 *)(params->cmxData + chunk_size_bytes);
        v_in_out0 = (half8 *)(params->cmxData + 2 * chunk_size_bytes);
        left_cmx -= 2 * chunk_size_bytes;
    }
    else if ((weights != 0) || (bias != 0))
    {
        // Make space for one of the two
        v_in_out0 = (half8 *)(params->cmxData + chunk_size_bytes);
        left_cmx -= chunk_size_bytes;
    }

    for (s32 chunk_i = 0; chunk_i < no_chunks; ++chunk_i)
    {
        int curr_chunk_offset_bytes = chunk_i * chunk_size_bytes;
        int real_chunk_size_bytes = MIN(chunk_size_bytes, width * sizeof(half) - curr_chunk_offset_bytes);
        int real_chunk_size_bl8h = (real_chunk_size_bytes + sizeof(half8) - 1) / sizeof(half8);

        if (weights != 0)
        {
            dmaTask.wait();
            dmaTask.start((uint8_t *)(weights) + curr_chunk_offset_bytes,
                        v_weights,
                        real_chunk_size_bytes);
            dmaTask.wait();
        }

        if (bias != 0)
        {
            dmaTask.wait();
            dmaTask.start(((u8*)bias) + curr_chunk_offset_bytes,
                        v_bias,
                        real_chunk_size_bytes);
        }

        u32 left_cmx_3_bl8h = ((left_cmx / sizeof(half8)) / 3);

        u32 n_steps = left_cmx_3_bl8h / real_chunk_size_bl8h;

    // NOTE: Pipelining is not supported on Kmb.
    // TODO: Investigate and enable this code
    //    if(n_steps > 0 && height > n_steps * 2)
    //     {
    //         half8 * v_in_out1 = v_in_out0 + left_cmx_3_bl8h;
    //         half8 * v_in_out2 = v_in_out0 + 2 * left_cmx_3_bl8h;
    //         u32 h = 0;
    //         int steps_to_process0 = n_steps;
    //         int steps_to_process1 = n_steps;

    //         {
    //             dma1.wait();
    //             dma1.start(((u8*)input) + curr_chunk_offset_bytes + h * in_stride_in_bytes,
    //                        v_in_out0,
    //                        real_chunk_size_bytes * steps_to_process0,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bytes,
    //                        in_stride_in_bytes,
    //                        real_chunk_size_bl8h * sizeof(half8));
    //         }

    //         h += steps_to_process0;
    //         {
    //             dma1.wait();
    //             dma1.start(((u8*)input) + curr_chunk_offset_bytes + h * in_stride_in_bytes,
    //                        v_in_out1,
    //                        real_chunk_size_bytes * steps_to_process1,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bytes,
    //                        in_stride_in_bytes,
    //                        real_chunk_size_bl8h * sizeof(half8));
    //             operation(v_in_out0, v_in_out0, v_weights, v_bias, steps_to_process0, real_chunk_size_bl8h, x, params->params);
    //         }

    //         h += steps_to_process1;
    //         for (; h < height; )
    //         {
    //             int steps_to_process2 = MIN(n_steps, height - h);
    //             dma1.wait();
    //             dma1.create(((u8*)input) + curr_chunk_offset_bytes + h * in_stride_in_bytes,
    //                        v_in_out2,
    //                        real_chunk_size_bytes * steps_to_process2,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bytes,
    //                        in_stride_in_bytes,
    //                        real_chunk_size_bl8h * sizeof(half8));

    //             dma2.create(v_in_out0,
    //                        ((u8*)output) + curr_chunk_offset_bytes + (h - steps_to_process1 - steps_to_process0) * out_stride_in_bytes,
    //                        real_chunk_size_bytes * steps_to_process0,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bl8h * sizeof(half8),
    //                        out_stride_in_bytes
    //                        );
    //             dma1.append(dma2);
    //             dma1.start();

    //             operation(v_in_out1, v_in_out1, v_weights, v_bias, steps_to_process1, real_chunk_size_bl8h, x, params->params);
    //             h += steps_to_process2;
    //             steps_to_process0 = steps_to_process1;
    //             steps_to_process1 = steps_to_process2;
    //             half8 * tmp = v_in_out0;
    //             v_in_out0 = v_in_out1;
    //             v_in_out1 = v_in_out2;
    //             v_in_out2 = tmp;
    //         }

    //         {
    //             dma1.wait();

    //             operation(v_in_out1, v_in_out1, v_weights, v_bias, steps_to_process1, real_chunk_size_bl8h, x, params->params);
    //             dma1.start(v_in_out0,
    //                        ((u8*)output) + curr_chunk_offset_bytes + (h - steps_to_process1 - steps_to_process0) * out_stride_in_bytes,
    //                        real_chunk_size_bytes * steps_to_process0,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bl8h * sizeof(half8),
    //                        out_stride_in_bytes
    //                        );
    //         }
    //         {
    //             dma1.wait();

    //             dma1.start(v_in_out1,
    //                        ((u8*)output) + curr_chunk_offset_bytes + (h - steps_to_process1) * out_stride_in_bytes,
    //                        real_chunk_size_bytes * steps_to_process1,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bytes,
    //                        real_chunk_size_bl8h * sizeof(half8),
    //                        out_stride_in_bytes
    //                        );
    //         }
    //     }
    //     else
        {
            n_steps = (left_cmx / sizeof(half8)) / real_chunk_size_bl8h;
            for (u32 h = 0; h < height; h += n_steps)
            {
                int steps_to_process = MIN(n_steps, height - h);
                dmaTask.start(((u8*)input) + curr_chunk_offset_bytes + h * in_stride_in_bytes,
                            v_in_out0,
                            real_chunk_size_bytes * steps_to_process,
                            real_chunk_size_bytes,
                            real_chunk_size_bytes,
                            in_stride_in_bytes,
                            real_chunk_size_bl8h * sizeof(half8));

                dmaTask.wait();

                operation(v_in_out0, v_in_out0, v_weights, v_bias, steps_to_process, real_chunk_size_bl8h, x, params->params);

                dmaTask.start(v_in_out0,
                            ((u8*)output) + curr_chunk_offset_bytes + h * out_stride_in_bytes,
                            real_chunk_size_bytes * steps_to_process,
                            real_chunk_size_bytes,
                            real_chunk_size_bytes,
                            real_chunk_size_bl8h * sizeof(half8),
                            out_stride_in_bytes
                            );
            }
        }
    }

    dmaTask.wait();
}

static void CHW_biasRelu_fp16(half8* __restrict__ in,
                              half8* __restrict__ out,
                              half* __restrict__ weights,
                              half* __restrict__ bias,
                              s32 channels,
                              s32 size,
                              half x,
                              void* params)
{
    UNUSED(weights);
    UNUSED(params);

    const half8  x_vec = (x <= half(0.0)) ? (half8)(X_MAX) : (half8)x;

    for (s32 c = 0; c < channels; ++c)
    {
        half8* __restrict__ rin = in + size * c;
        half8* __restrict__ rout = out + size * c;

        half8 rb0 = (half8) bias[c];

        s32 i = 0;

        if (size > 3)
        {
            half8 rr0 = rin[i + 0]; rr0 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr0 + rb0, x_vec);
            half8 rr1 = rin[i + 1]; rr1 = rr1 + rb0;
            half8 rr2 = rin[i + 2];

#pragma clang loop unroll_count(4)
            for (; i < size - 3; ++i)
            {
                half8 rr3 = rin[i + 3];
                rout[i] = rr0;
                rr0 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr1, x_vec);
                rr1 = rr2 + rb0;
                rr2 = rr3;
            }
        }

        for (; i < size; ++i)
        {
            rout[i] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[i] + rb0, x_vec);
        }
    }
}

static void CHW_biasLeakyRelu_fp16(half8* __restrict__ in,
                                   half8* __restrict__ out,
                                   half* __restrict__ weights,
                                   half* __restrict__ bias,
                                   s32 channels,
                                   s32 size,
                                   half x,
                                   void* params)
{
    UNUSED(weights);
    UNUSED(params);

    const half8 zeros = (half8) half(0.0f);
    const half8 x_vec = (half8)x;

    s32 c = 0;

    for (; c < channels - 1; c += 2)
    {
        half8* __restrict__ rin0 = in + size * (c + 0);
        half8* __restrict__ rin1 = in + size * (c + 1);
        half8* __restrict__ rout0 = out + size * (c + 0);
        half8* __restrict__ rout1 = out + size * (c + 1);

        half8 rb0 = (half8) bias[c + 0];
        half8 rb1 = (half8) bias[c + 1];

        s32 i = 0;

        if (size > 3)
        {
            half8 rr00 = rin0[i + 0];
            half8 rr01 = rin1[i + 0];
            rr00 = rr00 + rb0;
            rr01 = rr01 + rb1;
            half8 rq00 = __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros); rr00 = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros);
            half8 rq01 = __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros); rr01 = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros);

            half8 rr10 = rin0[i + 1];
            half8 rr11 = rin1[i + 1];
            rr10 = rr10 + rb0;
            rr11 = rr11 + rb1;

            half8 rr20 = rin0[i + 2];
            half8 rr21 = rin1[i + 2];

#pragma clang loop unroll_count(3)
            for (; i < size - 3; ++i)
            {
                half8 rr30 = rin0[i + 3];
                half8 rr31 = rin1[i + 3];
                rout0[i] = rr00 + rq00 * x_vec;
                rout1[i] = rr01 + rq01 * x_vec;
                rq00 = __builtin_shave_cmu_min_f16_rr_half8(rr10, zeros); rr00 = __builtin_shave_cmu_max_f16_rr_half8(rr10, zeros);
                rq01 = __builtin_shave_cmu_min_f16_rr_half8(rr11, zeros); rr01 = __builtin_shave_cmu_max_f16_rr_half8(rr11, zeros);
                rr10 = rr20 + rb0;
                rr11 = rr21 + rb1;
                rr20 = rr30;
                rr21 = rr31;
            }
        }

        for (; i < size; ++i)
        {
            half8 rr00 = rin0[i] + rb0;
            half8 rr01 = rin1[i] + rb1;
            rout0[i] = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros) * x_vec;
            rout1[i] = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros) * x_vec;
        }
    }

    for (; c < channels; ++c)
    {
        half8* __restrict__ rin = in + size * c;
        half8* __restrict__ rout = out + size * c;

        half8 rb0 = (half8) bias[c];

        s32 i = 0;

        if (size > 3)
        {
            half8 rr0 = rin[i + 0];
            rr0 = rr0 + rb0;
            half8 rq0 = __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros); rr0 = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros);

            half8 rr1 = rin[i + 1];
            rr1 = rr1 + rb0;

            half8 rr2 = rin[i + 2];

#pragma clang loop unroll_count(6)
            for (; i < size - 3; ++i)
            {
                half8 rr3 = rin[i + 3];
                rout[i] = rr0 + rq0 * x_vec;
                rq0 = __builtin_shave_cmu_min_f16_rr_half8(rr1, zeros); rr0 = __builtin_shave_cmu_max_f16_rr_half8(rr1, zeros);
                rr1 = rr2 + rb0;
                rr2 = rr3;
            }
        }

        for (; i < size; ++i)
        {
            half8 rr0 = rin[i] + rb0;
            rout[i] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros) * x_vec;
        }
    }
}

void CHW_prelu_fp16(half8* __restrict__ in,
                    half8* __restrict__ out,
                    half* __restrict__ weights,
                    half* __restrict__ bias,
                    s32 channels,
                    s32 size,
                    half x,
                    void* params)
{
    UNUSED(bias);
    UNUSED(x);
    UNUSED(params);

    const half8 zeros = (half8) half(0.0f);

    s32 c = 0;

    for (; c < channels - 1; c += 2)
    {
        half8* __restrict__ rin0 = in + size * (c + 0);
        half8* __restrict__ rin1 = in + size * (c + 1);
        half8* __restrict__ rout0 = out + size * (c + 0);
        half8* __restrict__ rout1 = out + size * (c + 1);

        half8 rw0 = (half8) weights[c + 0];
        half8 rw1 = (half8) weights[c + 1];

        s32 i = 0;

        if (size > 2)
        {
            half8 rr00 = rin0[i + 0];
            half8 rr01 = rin1[i + 0];
            half8 rq00 = __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros); rr00 = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros);
            half8 rq01 = __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);

            half8 rr10 = rin0[i + 1];
            half8 rr11 = rin1[i + 1];

#pragma clang loop unroll_count(3)
            for (; i < size - 2; ++i)
            {
                half8 rr20 = rin0[i + 2];
                half8 rr21 = rin1[i + 2];
                rout0[i] = rr00 + rq00 * rw0;
                rout1[i] = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + rq01 * rw1;
                rq00 = __builtin_shave_cmu_min_f16_rr_half8(rr10, zeros); rr00 = __builtin_shave_cmu_max_f16_rr_half8(rr10, zeros);
                rq01 = __builtin_shave_cmu_min_f16_rr_half8(rr11, zeros); rr01 = rr11;
                rr10 = rr20;
                rr11 = rr21;
            }
        }

        for (; i < size; ++i)
        {
            half8 rr0 = rin0[i];
            half8 rr1 = rin1[i];
            rout0[i] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros) * rw0;
            rout1[i] = __builtin_shave_cmu_max_f16_rr_half8(rr1, zeros) + __builtin_shave_cmu_min_f16_rr_half8(rr1, zeros) * rw1;
        }
    }

    for (; c < channels; ++c)
    {
        half8* __restrict__ rin = in + size * c;
        half8* __restrict__ rout = out + size * c;

        half8 rw0 = (half8) weights[c];

        s32 i = 0;

        if (size > 2)
        {
            half8 rr0 = rin[i + 0];
            half8 rq0 = __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros);

            half8 rr1 = rin[i + 1];

#pragma clang loop unroll_count(5)
            for (; i < size - 2; ++i)
            {
                half8 rr2 = rin[i + 2];
                rout[i] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + rq0 * rw0;
                rq0 = __builtin_shave_cmu_min_f16_rr_half8(rr1, zeros); rr0 = rr1;
                rr1 = rr2;
            }
        }

        for (; i < size; ++i)
        {
            half8 rr0 = rin[i];
            rout[i] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros) * rw0;
        }
    }
}

void CHW_bias_fp16(half8* __restrict__ in,
                        half8* __restrict__ out,
                        half* __restrict__ weights,
                        half* __restrict__ bias,
                        s32 channels,
                        s32 size,
                        half x,
                        void* params)
{
    UNUSED(weights);
    UNUSED(x);
    UNUSED(params);

    for (s32 c = 0; c < channels; ++c)
    {
        half8* __restrict__ rin = in + size * c;
        half8* __restrict__ rout = out + size * c;

        half8 rb0 = (half8) bias[c];

        s32 i = 0;

        if (size > 2)
        {
            half8 rr0 = rin[i + 0] + rb0;
            half8 rr1 = rin[i + 1];

#pragma clang loop unroll_count(6)
            for (; i < size - 2; ++i)
            {
                half8 rr2 = rin[i + 2];
                rout[i] = rr0;
                rr0 = rr1 + rb0;
                rr1 = rr2;
            }
        }

        for (; i < size; ++i)
        {
            rout[i] = rin[i] + rb0;
        }
    }
}

void CHW_scale_fp16(half8* __restrict__ in,
                        half8* __restrict__ out,
                        half* __restrict__ weights,
                        half* __restrict__ bias,
                        s32 channels,
                        s32 size,
                        half x,
                        void* params)
{
    UNUSED(bias);
    UNUSED(x);
    UNUSED(params);

    for (s32 c = 0; c < channels; ++c)
    {
        half8* __restrict__ rin = in + size * c;
        half8* __restrict__ rout = out + size * c;

        half8 rw0 = (half8) weights[c];

        s32 i = 0;

        if (size > 2)
        {
            half8 rr0 = rin[i + 0] * rw0;
            half8 rr1 = rin[i + 1];

#pragma clang loop unroll_count(6)
            for (; i < size - 2; ++i)
            {
                half8 rr2 = rin[i + 2];
                rout[i] = rr0;
                rr0 = rr1 * rw0;
                rr1 = rr2;
            }
        }

        for (; i < size; ++i)
        {
            rout[i] = rin[i] * rw0;
        }
    }
}

void CHW_scaleShift_fp16(half8* __restrict__ in,
                                half8* __restrict__ out,
                                half* __restrict__ weights,
                                half* __restrict__ bias,
                                s32 channels,
                                s32 size,
                                half x,
                                void* params)
{
    UNUSED(x);
    UNUSED(params);

    for (s32 c = 0; c < channels; ++c)
    {
        half8* __restrict__ rin = in + size * c;
        half8* __restrict__ rout = out + size * c;

        half8 rw0 = (half8) weights[c];
        half8 rb0 = (half8) bias[c];

        s32 i = 0;

        if (size > 3)
        {
            half8 rr0 = rin[i + 0] * rw0 + rb0;
            half8 rr1 = rin[i + 1] * rw0;
            half8 rr2 = rin[i + 2];

#pragma clang loop unroll_count(6)
            for (; i < size - 3; ++i)
            {
                half8 rr3 = rin[i + 3];
                rout[i] = rr0;
                rr0 = rr1 + rb0;
                rr1 = rr2 * rw0;
                rr2 = rr3;
            }
        }

        for (; i < size; ++i)
        {
            rout[i] = rin[i] * rw0 + rb0;
        }
    }
}

/* calculate the size for new line width */
int get_size_for_new_line_width(const int size, const int old_width, const int new_width)
{
    return ((size / old_width) * new_width + (size % old_width) % new_width);
}

bool getCHWOperation(t_PostOps postOpType, operation_chw_type &operation, half &p_x, void *p_params) {
    switch (postOpType)
    {
    case t_PostOps::BIAS_RELU:
        operation = &CHW_biasRelu_fp16;
        p_x = (half) *(float*)p_params;
        break;
    case t_PostOps::BIAS_LEAKY_RELU:
        operation = &CHW_biasLeakyRelu_fp16;
        p_x = (half) *(float*)p_params;
        break;
    case t_PostOps::PRELU:
        operation = &CHW_prelu_fp16;
        break;
    case t_PostOps::BIAS:
        operation = &CHW_bias_fp16;
        break;
    case t_PostOps::SCALE:
        operation = &CHW_scale_fp16;
        break;
    case t_PostOps::SCALE_SHIFT:
        operation = &CHW_scaleShift_fp16;
        break;
    case t_PostOps::CLAMP:
        operation = (operation_chw_type)(&clamp_fp16);
        break;
    case t_PostOps::ELU:
        operation = (operation_chw_type)(&eluFp16);
        p_x = (half) *(float*)p_params;
        break;
    case t_PostOps::POWER:
        operation = (operation_chw_type)(&power_fp16);
        break;
    case t_PostOps::SIGMOID:
        operation = (operation_chw_type)(&sigmoid_fp16);
        break;
    case t_PostOps::TANH:
        operation = (operation_chw_type)(&tanh_fp16);
        break;
    case t_PostOps::HSWISH:
        operation = (operation_chw_type)(&hswish_fp16);
        break;
    case t_PostOps::SWISH:
        operation = (operation_chw_type)(&swish_fp16);
        break;
    case t_PostOps::SOFTPLUS:
        operation = (operation_chw_type)(&softplus_fp16);
        break;
    case t_PostOps::MISH:
        operation = (operation_chw_type)(&mish_fp16);
        break;
    case t_PostOps::FLOOR:
        operation = (operation_chw_type)(&floor_fp16);
        break;
    case t_PostOps::CEIL:
        operation = (operation_chw_type)(&ceil_fp16);
        break;
    case t_PostOps::ROUND:
        operation = (operation_chw_type)(&round_fp16);
        break;
    case t_PostOps::ERF:
        operation = (operation_chw_type)(&erf_fp16);
        break;
    case t_PostOps::GELU:
        operation = (operation_chw_type)(&gelu_fp16);
        break;
    case t_PostOps::LOG:
        operation = (operation_chw_type)(&log_fp16);
        break;
    case t_PostOps::EXP:
        operation = (operation_chw_type)(&exp_fp16);
        break;
    default: return false;
    }

    return true;
}

void chw_postOps_3D_core(t_CHWPostOps3DParams *params)
{
    DmaAlShave dmaTask;

    const half* p_input   = params->input;
    half* p_output        = params->output;
    const half* p_weights = params->weights;
    const half* p_bias    = params->bias;
    s32 p_width           = params->width;
    s32 p_height          = params->height;
    s32 p_channels        = params->channels;
    s32 p_in_step         = params->in_step;
    s32 p_out_step        = params->out_step;
    half* p_cmxslice      = (half*)(params->cmxData);
    half p_x              = half(0);
    void* p_params        = params->params;

    operation_chw_type operation = nullptr;

    /* conditions for load/store without strides:
          input stride == output stride;
       && input stride % 8 == 0;
       && stride serves only for data alignment (input_stride - width <= 7)
    */
    bool load_store_with_inner_strides = !((p_in_step == p_out_step) && (p_in_step - p_width <= 7) && ((p_in_step & 7) == 0));

    bool success = getCHWOperation(params->postOpType, operation, p_x, p_params);
    if (!success) return;

    const auto CMX_SIZE = params->availableCmxBytes;

    half* cmx = p_cmxslice;
    s32 cmx_size = CMX_SIZE / sizeof(half);
    half* weights = cmx;
    if (p_weights != 0)
    {
        cmx += p_channels;
        cmx_size -= p_channels;

        dmaTask.wait();
        dmaTask.start(p_weights, weights, sizeof(half) * p_channels);
    }

    half* bias = cmx;
    if (p_bias != 0)
    {
        cmx += p_channels;
        cmx_size -= p_channels;

        dmaTask.wait();
        dmaTask.start(p_bias, bias, sizeof(half) * p_channels);

    }

    s32 width_8h = (p_width + 7) / 8;

    s32 plane_size_8h;
    if (load_store_with_inner_strides)
    {
        plane_size_8h = width_8h * p_height;
    }
    else
    {
        plane_size_8h = (p_in_step / 8) * p_height;
    }

    s32 cmx_size_8h = cmx_size / 8;

    s32 plane_divides = 1;
    s32 channels_per_chunk = cmx_size_8h / (plane_size_8h * 3);
    if (channels_per_chunk == 0)
    {
        plane_divides = (plane_size_8h * 3 + (cmx_size_8h - 1)) / cmx_size_8h;
        channels_per_chunk = 1;
    }

    s32 chunk_size_8h;
    if (plane_divides > 1)
    {
        chunk_size_8h = (plane_size_8h + (plane_divides - 1)) / plane_divides;

        // align the chunk size by the line size
        if (load_store_with_inner_strides)
            chunk_size_8h = (chunk_size_8h / (width_8h)) * (width_8h);
        else
            chunk_size_8h = (chunk_size_8h / (p_in_step / 8)) * (p_in_step / 8);

        if (chunk_size_8h <= 0)
        {
            if (params->ret_status)
                *(params->ret_status) = 1;
            return;
        }

        plane_divides = (plane_size_8h + (chunk_size_8h - 1)) / chunk_size_8h;
    }
    else
    {
        chunk_size_8h = plane_size_8h * channels_per_chunk;
    }

    s32 channel_divides = (p_channels + (channels_per_chunk - 1)) / channels_per_chunk;
    s32 total_chunks = channel_divides * plane_divides;

    half8* buf_0 = (half8*)cmx; cmx += 8 * chunk_size_8h;
//    half8* buf_1 = (half8*)cmx; cmx += 8 * chunk_size_8h;
//    half8* buf_2 = (half8*)cmx; //cmx += 8 * chunk_size_8h;

    s32 in_offset_0 = 0;
    s32 out_offset_0 = 0;
    s32 start_channel_0 = 0;

    s32 channels_to_process_0 = __builtin_shave_cmu_min_i32_rr_int(channels_per_chunk, p_channels - start_channel_0);
    s32 chunk_rest_8h = plane_size_8h * channels_to_process_0;

    // NOTE: Pipelining is not supported on Kmb.
    // TODO: Investigate and enable this code
    // if (total_chunks > 2)
    // {
    //     s32 size_to_process_8h_1, in_offset_1, out_offset_1, start_channel_1, channels_to_process_1;
    //     s32 size_to_process_8h_2, in_offset_2, out_offset_2, start_channel_2, channels_to_process_2;

    //     s32 chunk_i = 0;

    //     {
    //         s32 size_to_process_8h_0 = __builtin_shave_cmu_min_i32_rr_int(chunk_size_8h, chunk_rest_8h);

    //         dma1.wait();

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_load_store = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_width);

    //             dma1.start(p_input + in_offset_0, buf_0, sizeof(half) * size_to_load_store,

    //                                                      sizeof(half) * p_width,
    //                                                      sizeof(half) * p_width,

    //                                                      sizeof(half) * p_in_step,
    //                                                      sizeof(half) * 8 * width_8h);
    //         }
    //         else
    //         {
    //             dma1.start(p_input + in_offset_0, buf_0, sizeof(half) * 8 * size_to_process_8h_0);
    //         }

    //         half8* tmp = buf_0; buf_0 = buf_2; buf_2 = buf_1; buf_1 = tmp;

    //         in_offset_1 = in_offset_0;
    //         out_offset_1 = out_offset_0;
    //         start_channel_1 = start_channel_0;
    //         channels_to_process_1 = channels_to_process_0;
    //         size_to_process_8h_1 = size_to_process_8h_0;

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_in_offset = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_in_step);
    //             s32 size_to_out_offset = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_out_step);

    //             in_offset_0 += size_to_in_offset;
    //             out_offset_0 += size_to_out_offset;
    //         }
    //         else
    //         {
    //             in_offset_0 += 8 * size_to_process_8h_0;
    //             out_offset_0 += 8 * size_to_process_8h_0;
    //         }
    //         chunk_rest_8h -= size_to_process_8h_0;
    //         if (chunk_rest_8h <= 0)
    //         {
    //             start_channel_0 += channels_per_chunk;
    //             channels_to_process_0 = __builtin_shave_cmu_min_i32_rr_int(channels_per_chunk, p_channels - start_channel_0);
    //             chunk_rest_8h = plane_size_8h * channels_to_process_0;
    //         }
    //         ++chunk_i;
    //     }

    //     {
    //         s32 size_to_process_8h_0 = __builtin_shave_cmu_min_i32_rr_int(chunk_size_8h, chunk_rest_8h);

    //         dma1.wait();

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_load_store = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_width);

    //             dma1.start(p_input + in_offset_0, buf_0, sizeof(half) * size_to_load_store,

    //                                                      sizeof(half) * p_width,
    //                                                      sizeof(half) * p_width,

    //                                                      sizeof(half) * p_in_step,
    //                                                      sizeof(half) * 8 * width_8h);
    //         }
    //         else
    //         {
    //             dma1.start(p_input + in_offset_0, buf_0, sizeof(half) * 8 * size_to_process_8h_0);
    //         }

    //         s32 calc_size_8h = __builtin_shave_cmu_min_i32_rr_int(size_to_process_8h_1, plane_size_8h);
    //         operation(buf_1, buf_1, weights + start_channel_1, bias + start_channel_1, channels_to_process_1, calc_size_8h, p_x, p_params);

    //         half8* tmp = buf_0; buf_0 = buf_2; buf_2 = buf_1; buf_1 = tmp;

    //         in_offset_2 = in_offset_1;
    //         out_offset_2 = out_offset_1;
    //         start_channel_2 = start_channel_1;
    //         channels_to_process_2 = channels_to_process_1;
    //         size_to_process_8h_2 = size_to_process_8h_1;

    //         in_offset_1 = in_offset_0;
    //         out_offset_1 = out_offset_0;
    //         start_channel_1 = start_channel_0;
    //         channels_to_process_1 = channels_to_process_0;
    //         size_to_process_8h_1 = size_to_process_8h_0;

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_in_offset = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_in_step);
    //             s32 size_to_out_offset = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_out_step);

    //             in_offset_0 += size_to_in_offset;
    //             out_offset_0 += size_to_out_offset;
    //         }
    //         else
    //         {
    //             in_offset_0 += 8 * size_to_process_8h_0;
    //             out_offset_0 += 8 * size_to_process_8h_0;
    //         }
    //         chunk_rest_8h -= size_to_process_8h_0;
    //         if (chunk_rest_8h <= 0)
    //         {
    //             start_channel_0 += channels_per_chunk;
    //             channels_to_process_0 = __builtin_shave_cmu_min_i32_rr_int(channels_per_chunk, p_channels - start_channel_0);
    //             chunk_rest_8h = plane_size_8h * channels_to_process_0;
    //         }
    //         ++chunk_i;
    //     }

    //     for (; chunk_i < total_chunks; )
    //     {
    //         s32 size_to_process_8h_0 = __builtin_shave_cmu_min_i32_rr_int(chunk_size_8h, chunk_rest_8h);

    //         dma1.wait();
    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_load_store = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_width);

    //             dma1.create(p_input + in_offset_0, buf_0, sizeof(half) * size_to_load_store,

    //                                                       sizeof(half) * p_width,
    //                                                       sizeof(half) * p_width,

    //                                                       sizeof(half) * p_in_step,
    //                                                       sizeof(half) * 8 * width_8h);
    //         }
    //         else
    //         {
    //             dma1.create(p_input + in_offset_0, buf_0, sizeof(half) * 8 * size_to_process_8h_0);
    //         }

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_store_store = get_size_for_new_line_width(8 * size_to_process_8h_2, width_8h * 8, p_width);

    //             dma2.create(buf_2, p_output + out_offset_2, sizeof(half) * size_to_store_store,

    //                                                         sizeof(half) * p_width,
    //                                                         sizeof(half) * p_width,

    //                                                         sizeof(half) * 8 * width_8h,
    //                                                         sizeof(half) * p_out_step);
    //         }
    //         else
    //         {
    //             dma2.create(buf_2, p_output + out_offset_2, sizeof(half) * 8 * size_to_process_8h_2);
    //         }

    //         dma1.append(dma2);
    //         dma1.start();

    //         s32 calc_size_8h = __builtin_shave_cmu_min_i32_rr_int(size_to_process_8h_1, plane_size_8h);
    //         operation(buf_1, buf_1, weights + start_channel_1, bias + start_channel_1, channels_to_process_1, calc_size_8h, p_x, p_params);

    //         half8* tmp = buf_0; buf_0 = buf_2; buf_2 = buf_1; buf_1 = tmp;

    //         in_offset_2 = in_offset_1;
    //         out_offset_2 = out_offset_1;
    //         start_channel_2 = start_channel_1;
    //         channels_to_process_2 = channels_to_process_1;
    //         size_to_process_8h_2 = size_to_process_8h_1;

    //         in_offset_1 = in_offset_0;
    //         out_offset_1 = out_offset_0;
    //         start_channel_1 = start_channel_0;
    //         channels_to_process_1 = channels_to_process_0;
    //         size_to_process_8h_1 = size_to_process_8h_0;

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_in_offset = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_in_step);
    //             s32 size_to_out_offset = get_size_for_new_line_width(8 * size_to_process_8h_0, width_8h * 8, p_out_step);

    //             in_offset_0 += size_to_in_offset;
    //             out_offset_0 += size_to_out_offset;
    //         }
    //         else
    //         {
    //             in_offset_0 += 8 * size_to_process_8h_0;
    //             out_offset_0 += 8 * size_to_process_8h_0;
    //         }
    //         chunk_rest_8h -= size_to_process_8h_0;
    //         if (chunk_rest_8h <= 0)
    //         {
    //             start_channel_0 += channels_per_chunk;
    //             channels_to_process_0 = __builtin_shave_cmu_min_i32_rr_int(channels_per_chunk, p_channels - start_channel_0);
    //             chunk_rest_8h = plane_size_8h * channels_to_process_0;
    //         }
    //         ++chunk_i;
    //     }

    //     {
    //         dma1.wait();

    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_store_store = get_size_for_new_line_width(8 * size_to_process_8h_2, width_8h * 8, p_width);

    //             dma1.start(buf_2, p_output + out_offset_2, sizeof(half) * size_to_store_store,

    //                                                        sizeof(half) * p_width,
    //                                                        sizeof(half) * p_width,

    //                                                        sizeof(half) * 8 * width_8h,
    //                                                        sizeof(half) * p_out_step);
    //         }
    //         else
    //         {
    //             dma1.start(buf_2, p_output + out_offset_2, sizeof(half) * 8 * size_to_process_8h_2);
    //         }

    //         s32 calc_size_8h = __builtin_shave_cmu_min_i32_rr_int(size_to_process_8h_1, plane_size_8h);
    //         s32 calc_size_8h = __builtin_shave_cmu_min_i32_rr_int(size_to_process_8h_1, plane_size_8h);
    //         operation(buf_1, buf_1, weights + start_channel_1, bias + start_channel_1, channels_to_process_1, calc_size_8h, p_x, p_params);

    //         buf_2 = buf_1;

    //         in_offset_2 = in_offset_1;
    //         out_offset_2 = out_offset_1;
    //         size_to_process_8h_2 = size_to_process_8h_1;
    //     }

    //     {
    //         dma1.wait();
    //         if (load_store_with_inner_strides)
    //         {
    //             s32 size_to_store_store = get_size_for_new_line_width(8 * size_to_process_8h_2, width_8h * 8, p_width);

    //             dma1.start(buf_2, p_output + out_offset_2, sizeof(half) * size_to_store_store,

    //                                                        sizeof(half) * p_width,
    //                                                        sizeof(half) * p_width,

    //                                                        sizeof(half) * 8 * width_8h,
    //                                                        sizeof(half) * p_out_step);
    //         }
    //         else
    //         {
    //             dma1.start(buf_2, p_output + out_offset_2, sizeof(half) * 8 * size_to_process_8h_2);
    //         }
    //     }
    // } else
    {
        for (s32 chunk_i = 0; chunk_i < total_chunks; ++chunk_i)
        {
            s32 size_to_process_8h_0 = __builtin_shave_cmu_min_i32_rr_int(chunk_size_8h, chunk_rest_8h);
            s32 size_to_process = size_to_process_8h_0 * 8;

            dmaTask.wait();

            if (load_store_with_inner_strides)
            {
                s32 size_to_load_store = get_size_for_new_line_width(size_to_process, width_8h * 8, p_width);

                dmaTask.start(p_input + in_offset_0, buf_0, sizeof(half) * size_to_load_store,

                                                         sizeof(half) * p_width,
                                                         sizeof(half) * p_width,

                                                         sizeof(half) * p_in_step,
                                                         sizeof(half) * 8 * width_8h);
            }
            else
            {
                dmaTask.start(p_input + in_offset_0, buf_0, sizeof(half) * 8 * size_to_process_8h_0);
            }

            dmaTask.wait();
            s32 calc_size_8h = __builtin_shave_cmu_min_i32_rr_int(size_to_process_8h_0, plane_size_8h);
            operation(buf_0, buf_0, weights + start_channel_0, bias + start_channel_0, channels_to_process_0, calc_size_8h, p_x, p_params);

            if (load_store_with_inner_strides)
            {
                s32 size_to_store_store = get_size_for_new_line_width(size_to_process, width_8h * 8, p_width);

                dmaTask.start(buf_0, p_output + out_offset_0, sizeof(half) * size_to_store_store,

                                                           sizeof(half) * p_width,
                                                           sizeof(half) * p_width,

                                                           sizeof(half) * 8 * width_8h,
                                                           sizeof(half) * p_out_step);
            }
            else
            {
                dmaTask.start(buf_0, p_output + out_offset_0, sizeof(half) * 8 * size_to_process_8h_0);
            }

            if (load_store_with_inner_strides)
            {
                s32 size_to_in_offset = get_size_for_new_line_width(size_to_process, width_8h * 8, p_in_step);
                s32 size_to_out_offset = get_size_for_new_line_width(size_to_process, width_8h * 8, p_out_step);

                in_offset_0 += size_to_in_offset;
                out_offset_0 += size_to_out_offset;
            }
            else
            {
                in_offset_0 += size_to_process;
                out_offset_0 += size_to_process;
            }

            chunk_rest_8h -= size_to_process_8h_0;
            if (chunk_rest_8h <= 0)
            {
                start_channel_0 += channels_per_chunk;
                channels_to_process_0 = __builtin_shave_cmu_min_i32_rr_int(channels_per_chunk, p_channels - start_channel_0);
                chunk_rest_8h = plane_size_8h * channels_to_process_0;
            }
        }
    }
}

struct InterleavedPostOpParameters
{
    unsigned depth;
    unsigned current_input_line;
    void *params;
};

void HCW_prelu_fp16(half8 * data_in,
                    half8 * data_out,
                    half8 * weights,
                    half8 *,
                    s32 no_lines, s32 line_size, half,
                    void *parameters)
{
    InterleavedPostOpParameters *p = reinterpret_cast<InterleavedPostOpParameters *>(parameters);
    unsigned current_input_line = p->current_input_line;
    unsigned depth = p->depth;
    const half8 zeros = (half8)0.0;

    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {

        half8 channel_weights[8];

        channel_weights[0] = *((half*)weights + (current_input_line + k + 0) % depth);
        channel_weights[1] = *((half*)weights + (current_input_line + k + 1) % depth);
        channel_weights[2] = *((half*)weights + (current_input_line + k + 2) % depth);
        channel_weights[3] = *((half*)weights + (current_input_line + k + 3) % depth);
        channel_weights[4] = *((half*)weights + (current_input_line + k + 4) % depth);
        channel_weights[5] = *((half*)weights + (current_input_line + k + 5) % depth);
        channel_weights[6] = *((half*)weights + (current_input_line + k + 6) % depth);
        channel_weights[7] = *((half*)weights + (current_input_line + k + 7) % depth);

        for (int j = 0; j < line_size; j++)
        {
            data_out[(k + 0) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 0) * line_size + j],zeros) + channel_weights[0] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 0) * line_size + j],zeros);
            data_out[(k + 1) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 1) * line_size + j],zeros) + channel_weights[1] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 1) * line_size + j],zeros);
            data_out[(k + 2) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 2) * line_size + j],zeros) + channel_weights[2] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 2) * line_size + j],zeros);
            data_out[(k + 3) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 3) * line_size + j],zeros) + channel_weights[3] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 3) * line_size + j],zeros);
            data_out[(k + 4) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 4) * line_size + j],zeros) + channel_weights[4] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 4) * line_size + j],zeros);
            data_out[(k + 5) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 5) * line_size + j],zeros) + channel_weights[5] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 5) * line_size + j],zeros);
            data_out[(k + 6) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 6) * line_size + j],zeros) + channel_weights[6] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 6) * line_size + j],zeros);
            data_out[(k + 7) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 7) * line_size + j],zeros) + channel_weights[7] * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 7) * line_size + j],zeros);
        }
    }

}

void HCW_biasRelu_fp16(half8 * data_in,
                       half8 * data_out,
                       half8 *,
                       half8 * bias_,
                       s32 no_lines, s32 line_size, half x,
                       void *parameters)
{
    InterleavedPostOpParameters *p = reinterpret_cast<InterleavedPostOpParameters *>(parameters);
    unsigned current_input_line = p->current_input_line;
    unsigned depth = p->depth;
    half *bias = (half *) bias_;

    const half8 x_vec = (x <= half(0.0)) ? (half8)(X_MAX) : (half8)x;

    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {
        for (int j = 0; j < line_size; j++)
        {
            data_out[(k + 0) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 0) * line_size + j] + (half8)(bias[(current_input_line + k + 0) % depth]), x_vec);
            data_out[(k + 1) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 1) * line_size + j] + (half8)(bias[(current_input_line + k + 1) % depth]), x_vec);
            data_out[(k + 2) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 2) * line_size + j] + (half8)(bias[(current_input_line + k + 2) % depth]), x_vec);
            data_out[(k + 3) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 3) * line_size + j] + (half8)(bias[(current_input_line + k + 3) % depth]), x_vec);
            data_out[(k + 4) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 4) * line_size + j] + (half8)(bias[(current_input_line + k + 4) % depth]), x_vec);
            data_out[(k + 5) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 5) * line_size + j] + (half8)(bias[(current_input_line + k + 5) % depth]), x_vec);
            data_out[(k + 6) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 6) * line_size + j] + (half8)(bias[(current_input_line + k + 6) % depth]), x_vec);
            data_out[(k + 7) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 7) * line_size + j] + (half8)(bias[(current_input_line + k + 7) % depth]), x_vec);
        }
    }
}

void HCW_biasLeakyRelu_fp16(half8 * data_in,
                            half8 * data_out,
                            half8 *,
                            half8 * bias_,
                            s32 no_lines, s32 line_size, half x,
                            void *parameters)
{
    InterleavedPostOpParameters *p = reinterpret_cast<InterleavedPostOpParameters *>(parameters);
    unsigned current_input_line = p->current_input_line;
    unsigned depth = p->depth;
    half *bias = (half *) bias_;

    const half8 zeros = (half8) half(0.0f);
    const half8 x_vec = (half8)x;

    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {
        for (int j = 0; j < line_size; j++)
        {

            data_out[(k + 0) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 0) * line_size + j] + (half8)(bias[(current_input_line + k + 0) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 0) * line_size + j] + (half8)(bias[(current_input_line + k + 0) % depth]), zeros);
            data_out[(k + 1) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 1) * line_size + j] + (half8)(bias[(current_input_line + k + 1) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 1) * line_size + j] + (half8)(bias[(current_input_line + k + 1) % depth]), zeros);
            data_out[(k + 2) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 2) * line_size + j] + (half8)(bias[(current_input_line + k + 2) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 2) * line_size + j] + (half8)(bias[(current_input_line + k + 2) % depth]), zeros);
            data_out[(k + 3) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 3) * line_size + j] + (half8)(bias[(current_input_line + k + 3) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 3) * line_size + j] + (half8)(bias[(current_input_line + k + 3) % depth]), zeros);
            data_out[(k + 4) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 4) * line_size + j] + (half8)(bias[(current_input_line + k + 4) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 4) * line_size + j] + (half8)(bias[(current_input_line + k + 4) % depth]), zeros);
            data_out[(k + 5) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 5) * line_size + j] + (half8)(bias[(current_input_line + k + 5) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 5) * line_size + j] + (half8)(bias[(current_input_line + k + 5) % depth]), zeros);
            data_out[(k + 6) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 6) * line_size + j] + (half8)(bias[(current_input_line + k + 6) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 6) * line_size + j] + (half8)(bias[(current_input_line + k + 6) % depth]), zeros);
            data_out[(k + 7) * line_size + j] = __builtin_shave_cmu_max_f16_rr_half8(data_in[(k + 7) * line_size + j] + (half8)(bias[(current_input_line + k + 7) % depth]), zeros) + x_vec * __builtin_shave_cmu_min_f16_rr_half8(data_in[(k + 7) * line_size + j] + (half8)(bias[(current_input_line + k + 7) % depth]), zeros);
        }
    }
}

void HCW_relu_fp16(half8 * data_in,
                   half8 * data_out,
                   half8 *,
                   half8 *,
                   s32 no_lines, s32 line_size, half x,
                   void *)
{
    const half8 x_vec = (x <= half(0.0)) ? (half8)(X_MAX) : (half8)x;

    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {
        for (int j = 0; j < line_size; j++)
        {
            data_out[(k + 0) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 0) * line_size + j], x_vec);
            data_out[(k + 1) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 1) * line_size + j], x_vec);
            data_out[(k + 2) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 2) * line_size + j], x_vec);
            data_out[(k + 3) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 3) * line_size + j], x_vec);
            data_out[(k + 4) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 4) * line_size + j], x_vec);
            data_out[(k + 5) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 5) * line_size + j], x_vec);
            data_out[(k + 6) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 6) * line_size + j], x_vec);
            data_out[(k + 7) * line_size + j] = __builtin_shave_cmu_clamp0_f16_rr_half8(data_in[(k + 7) * line_size + j], x_vec);
        }
    }
}


void HCW_bias_fp16(half8 * data_in,
                         half8 * data_out,
                         half8 *,
                         half8 * bias_,
                         s32 no_lines, s32 line_size, half,
                         void *parameters)
{
    InterleavedPostOpParameters *p = reinterpret_cast<InterleavedPostOpParameters *>(parameters);
    unsigned current_input_line = p->current_input_line;
    unsigned depth = p->depth;
    half *bias = (half *) bias_;

    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {
        for (int j = 0; j < line_size; j++)
        {
            data_out[(k + 0) * line_size + j] = data_in[(k + 0) * line_size + j] + (half8)(bias[(current_input_line + k + 0) % depth]);
            data_out[(k + 1) * line_size + j] = data_in[(k + 1) * line_size + j] + (half8)(bias[(current_input_line + k + 1) % depth]);
            data_out[(k + 2) * line_size + j] = data_in[(k + 2) * line_size + j] + (half8)(bias[(current_input_line + k + 2) % depth]);
            data_out[(k + 3) * line_size + j] = data_in[(k + 3) * line_size + j] + (half8)(bias[(current_input_line + k + 3) % depth]);
            data_out[(k + 4) * line_size + j] = data_in[(k + 4) * line_size + j] + (half8)(bias[(current_input_line + k + 4) % depth]);
            data_out[(k + 5) * line_size + j] = data_in[(k + 5) * line_size + j] + (half8)(bias[(current_input_line + k + 5) % depth]);
            data_out[(k + 6) * line_size + j] = data_in[(k + 6) * line_size + j] + (half8)(bias[(current_input_line + k + 6) % depth]);
            data_out[(k + 7) * line_size + j] = data_in[(k + 7) * line_size + j] + (half8)(bias[(current_input_line + k + 7) % depth]);
        }
    }
}

void HCW_scale_fp16(half8 * data_in,
                          half8 * data_out,
                          half8 * weights,
                          half8 *,
                          s32 no_lines, s32 line_size, half,
                          void *parameters)
{
    InterleavedPostOpParameters *p = reinterpret_cast<InterleavedPostOpParameters *>(parameters);
    unsigned current_input_line = p->current_input_line;
    unsigned depth = p->depth;

    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {
        half8 channel_weights[8];

        channel_weights[0] = *((half*)weights + (current_input_line + k + 0) % depth);
        channel_weights[1] = *((half*)weights + (current_input_line + k + 1) % depth);
        channel_weights[2] = *((half*)weights + (current_input_line + k + 2) % depth);
        channel_weights[3] = *((half*)weights + (current_input_line + k + 3) % depth);
        channel_weights[4] = *((half*)weights + (current_input_line + k + 4) % depth);
        channel_weights[5] = *((half*)weights + (current_input_line + k + 5) % depth);
        channel_weights[6] = *((half*)weights + (current_input_line + k + 6) % depth);
        channel_weights[7] = *((half*)weights + (current_input_line + k + 7) % depth);

        for (int j = 0; j < line_size; j++)
        {
            data_out[(k + 0) * line_size + j] = data_in[(k + 0) * line_size + j] * channel_weights[0];
            data_out[(k + 1) * line_size + j] = data_in[(k + 1) * line_size + j] * channel_weights[1];
            data_out[(k + 2) * line_size + j] = data_in[(k + 2) * line_size + j] * channel_weights[2];
            data_out[(k + 3) * line_size + j] = data_in[(k + 3) * line_size + j] * channel_weights[3];
            data_out[(k + 4) * line_size + j] = data_in[(k + 4) * line_size + j] * channel_weights[4];
            data_out[(k + 5) * line_size + j] = data_in[(k + 5) * line_size + j] * channel_weights[5];
            data_out[(k + 6) * line_size + j] = data_in[(k + 6) * line_size + j] * channel_weights[6];
            data_out[(k + 7) * line_size + j] = data_in[(k + 7) * line_size + j] * channel_weights[7];
        }
    }
}

void HCW_scaleShift_fp16(half8 * data_in,
                                  half8 * data_out,
                                  half8 * weights,
                                  half8 * bias,
                                  s32 no_lines, s32 line_size, half,
                                  void *parameters)
{
    InterleavedPostOpParameters *p = reinterpret_cast<InterleavedPostOpParameters *>(parameters);
    unsigned current_input_line = p->current_input_line;
    unsigned depth = p->depth;
    for (int k = 0; k < ((no_lines + 7) & ~0x07); k += 8)
    {
        half8 channel_weights[8];
        half8 channel_bias[8];

        channel_weights[0] = *((half*)weights + (current_input_line + k + 0) % depth);
        channel_weights[1] = *((half*)weights + (current_input_line + k + 1) % depth);
        channel_weights[2] = *((half*)weights + (current_input_line + k + 2) % depth);
        channel_weights[3] = *((half*)weights + (current_input_line + k + 3) % depth);
        channel_weights[4] = *((half*)weights + (current_input_line + k + 4) % depth);
        channel_weights[5] = *((half*)weights + (current_input_line + k + 5) % depth);
        channel_weights[6] = *((half*)weights + (current_input_line + k + 6) % depth);
        channel_weights[7] = *((half*)weights + (current_input_line + k + 7) % depth);

        channel_bias[0] = *((half*)bias + (current_input_line + k + 0) % depth);
        channel_bias[1] = *((half*)bias + (current_input_line + k + 1) % depth);
        channel_bias[2] = *((half*)bias + (current_input_line + k + 2) % depth);
        channel_bias[3] = *((half*)bias + (current_input_line + k + 3) % depth);
        channel_bias[4] = *((half*)bias + (current_input_line + k + 4) % depth);
        channel_bias[5] = *((half*)bias + (current_input_line + k + 5) % depth);
        channel_bias[6] = *((half*)bias + (current_input_line + k + 6) % depth);
        channel_bias[7] = *((half*)bias + (current_input_line + k + 7) % depth);

        for (int j = 0; j < line_size; j++)
        {
            data_out[(k + 0) * line_size + j] = data_in[(k + 0) * line_size + j] * channel_weights[0] + channel_bias[0];
            data_out[(k + 1) * line_size + j] = data_in[(k + 1) * line_size + j] * channel_weights[1] + channel_bias[1];
            data_out[(k + 2) * line_size + j] = data_in[(k + 2) * line_size + j] * channel_weights[2] + channel_bias[2];
            data_out[(k + 3) * line_size + j] = data_in[(k + 3) * line_size + j] * channel_weights[3] + channel_bias[3];
            data_out[(k + 4) * line_size + j] = data_in[(k + 4) * line_size + j] * channel_weights[4] + channel_bias[4];
            data_out[(k + 5) * line_size + j] = data_in[(k + 5) * line_size + j] * channel_weights[5] + channel_bias[5];
            data_out[(k + 6) * line_size + j] = data_in[(k + 6) * line_size + j] * channel_weights[6] + channel_bias[6];
            data_out[(k + 7) * line_size + j] = data_in[(k + 7) * line_size + j] * channel_weights[7] + channel_bias[7];
        }
    }
}

bool getHWCOperation(const t_PostOps postOpType, operation_hwc_type &operation, half &x, void *pparams) {
    switch (postOpType)
    {
    case t_PostOps::RELU:
        operation = &relu_fp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::LEAKY_RELU:
        operation = &leakyRelu_fp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::BIAS_RELU:
        operation = &HWC_biasRelu_fp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::BIAS_LEAKY_RELU:
        operation = &HWC_biasLeakyRelu_fp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::PRELU:
        operation = &HWC_prelu_fp16;
        return true;
    case t_PostOps::BIAS:
        operation = &HWC_bias_fp16;
        return true;
    case t_PostOps::SCALE:
        operation = &HWC_scale_fp16;
        return true;
    case t_PostOps::SCALE_SHIFT:
        operation = &HWC_scaleShift_fp16;
        return true;
    case t_PostOps::CLAMP:
        operation = (operation_hwc_type)(&clamp_fp16);
        return true;
    case t_PostOps::ELU:
        operation = &eluFp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::POWER:
        operation = &power_fp16;
        return true;
    case t_PostOps::SIGMOID:
        operation = (operation_hwc_type)(&sigmoid_fp16);
        return true;
    case t_PostOps::TANH:
        operation = (operation_hwc_type)(&tanh_fp16);
        return true;
    case t_PostOps::HSWISH:
        operation = (operation_hwc_type)(&hswish_fp16);
        return true;
    case t_PostOps::SWISH:
        operation = (operation_hwc_type)(&swish_fp16);
        return true;
    case t_PostOps::SOFTPLUS:
        operation = (operation_hwc_type)(&softplus_fp16);
        return true;
    case t_PostOps::MISH:
        operation = (operation_hwc_type)(&mish_fp16);
        return true;
    case t_PostOps::FLOOR:
        operation = (operation_hwc_type)(&floor_fp16);
        return true;
    case t_PostOps::CEIL:
        operation = (operation_hwc_type)(&ceil_fp16);
        return true;
    case t_PostOps::ROUND:
        operation = (operation_hwc_type)(&round_fp16);
        return true;
    case t_PostOps::ERF:
        operation = (operation_hwc_type)(&erf_fp16);
        return true;
    case t_PostOps::EXP:
        operation = (operation_hwc_type)(&exp_fp16);
        return true;
    case t_PostOps::GELU:
        operation = (operation_hwc_type)(&gelu_fp16);
        return true;
    case t_PostOps::LOG:
        operation = (operation_hwc_type)(&log_fp16);
        return true;
    default: return false;
    }
}

bool getHCWOperation(const t_PostOps postOpType, operation_hcw_type &operation, half &x, void *pparams) {
    switch(postOpType)
    {
    case t_PostOps::RELU:
        operation = &HCW_relu_fp16;
        x = half(0.0);
        return true;
    case t_PostOps::BIAS_RELU:
        operation = &HCW_biasRelu_fp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::BIAS_LEAKY_RELU:
        operation = &HCW_biasLeakyRelu_fp16;
        x = (half) *(float*)pparams;
        return true;
    case t_PostOps::PRELU:
        operation = &HCW_prelu_fp16;
        return true;
    case t_PostOps::BIAS:
        operation = &HCW_bias_fp16;
        return true;
    case t_PostOps::SCALE:
        operation = &HCW_scale_fp16;
        return true;
    case t_PostOps::SCALE_SHIFT:
        operation = &HCW_scaleShift_fp16;
        return true;
    default: return false;
    }
}

void hcw_postOps_3D_core(t_HCWPostOps3DParams *params)
{
    DmaAlShave dmaTask;

    const half *input   = params->input - params->offset;
    half *output        = params->output - params->offset;
    const half *weights = params->weights;
    const half *bias    = params->bias;
    u32  width          = params->width;
    u32  depth          = params->channels;
    u32  input_stride   = params->in_step;
    u32  output_stride  = params->out_step;
    half x              = half(0);
    void *pparams       = params->params;

    operation_hcw_type operation = nullptr;

    bool success = getHCWOperation(params->postOpType, operation, x, pparams);
    if (!success) return;

    // Interleaved format is used by hardware which means that the width
    // is aligned to 8 elements.

    // Each shave is given a different starting line in the range [start_line, assigned_lines)
    s32 start_line = params->start_line;

    // The number of lines the current shave needs to process
    s32 assigned_lines = params->height;

    const auto CMX_SIZE = params->availableCmxBytes;
    s32 cmx_bl8h    = (CMX_SIZE >> 3) / sizeof(half); // Size of cmx in blocks of 8 halfs.
    s32 w_bl8h      = width >> 3; // Size of width in blocks of 8 halfs.
    s32 w_remainder = width % 8;  // Number of halfs in the last block of 8 halfs.

    // "Padd" width to a round number of blocks of 8 halfs.
    if(w_remainder)
        ++w_bl8h; // There is an incomplete block of 8 halfs.

    // Round up to next multiple of 8 and calculate the bias in blocks of 8 halfs
    s32 b_bl8h = ((depth + 7) & ~0x07) >> 3;

    // Check if the whole bias and / or weights can fit in CMX
    if((cmx_bl8h - (!!bias + !!weights) * b_bl8h) < 0)
        return;

    half8 *v_bias = nullptr;
    half8 *v_weights = nullptr;

    if(bias){
        v_bias = (half8 *)(params->cmxData);
        // Bring the bias
        dmaTask.start(
        (u8 *)(bias),
        (u8 *)v_bias,
        (b_bl8h << 3) * sizeof(half));
    }

    if(weights){
        v_weights = bias ? (half8 *)(params->cmxData + (b_bl8h << 3) * sizeof(half)) : (half8 *)(params->cmxData);;
        // Bring the weights
        dmaTask.start(
        (u8 *)(weights),
        (u8 *)v_weights,
        (b_bl8h << 3) * sizeof(half));
    }

    // Make space for working buffer
    half8 *v_in_out = (half8 *)(params->cmxData);
    v_in_out += (!!bias + !!weights) * b_bl8h;

    // Compute how many lines of the input we can fit in CMX
    s32 no_cmx_8lines = 0;
    no_cmx_8lines = (cmx_bl8h - (!!bias + !!weights) * b_bl8h) / w_bl8h / 8;
    // Check if at least 8 whole lines of aligned width can fit in CMX
    if(no_cmx_8lines <= 0 )
        return;

    for (int i = 0; i < assigned_lines; i += (no_cmx_8lines << 3))
    {
        int start_idx = i;
        int end_idx = start_idx + MIN((no_cmx_8lines << 3), assigned_lines - i);

        int transfer_in_size = (end_idx - start_idx) * width * sizeof(half);
        int transfer_out_size = transfer_in_size;

        // TODO: Memcpy if very small
        dmaTask.start(
                (u8 *)(input + (start_line + i) * input_stride),
                (u8 *)v_in_out,
                transfer_in_size,
                width * sizeof(half),
                width * sizeof(half),
                input_stride * sizeof(half),
                (w_bl8h << 3) * sizeof(half));
        // Determine where to start in bias
        u32 current_input_line = start_line + i;
        InterleavedPostOpParameters p = {depth, current_input_line, params->params};
        operation(v_in_out, v_in_out, v_weights, v_bias, end_idx - start_idx, w_bl8h, x, &p);

        // TODO: Memcpy if very small
        dmaTask.start(
                (u8 *)(v_in_out),
                (u8 *)(output + (start_line + i) * output_stride),
                transfer_out_size,
                width * sizeof(half),
                width * sizeof(half),
                (w_bl8h << 3) * sizeof(half),
                output_stride * sizeof(half));
    }
}
