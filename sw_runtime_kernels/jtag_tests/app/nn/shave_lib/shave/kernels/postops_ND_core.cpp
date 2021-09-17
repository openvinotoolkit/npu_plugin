// {% copyright %}

#include <math.h>
#include <param_postops.h>

#include <dma_shave.h>
#include <sw_shave_lib_common.h>
#include <moviVectorConvert.h>

#include <svuCommonShave.h>
#include <nn_log.h>

using namespace nn::shave_lib;

//#define ASYNC_PIPELINE /* TODO: fully remove async code, if async pipelining isn't supported */

constexpr int INPUT_BPP = 2; // sizeof(fp16)

#define X_MAX       (65000)
#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

// Tools older than 00.50.79.2 have a different define
#ifndef pows
  #define pows(a, b) __hpow(a, b)
#endif

#define MIN(_a, _b) (__builtin_shave_cmu_min_i32_rr_int((_a), (_b)))
#define MAX(_a, _b) (__builtin_shave_cmu_max_i32_rr_int((_a), (_b)))

#define DIVR(_val, _size) (((_val) + ((_size) - 1)) / (_size))
#define ALIGN_TO_MULTIPLE(_size, _val) (DIVR((_val), (_size)) * (_size))

#define SWAP3(_p0, _p1, _p2) do { auto tmp = (_p2); (_p2) = (_p1); (_p1) = (_p0); (_p0) = tmp; } while(0)

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

typedef void (*Operation)(const half8*, half8*, const half8*, const half8*, int, int, int, const PostOpsNDParams*);
typedef void (*Pipeline)(Operation, const PostOpsNDParams*);

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

/* vector implementation of library roundf(x) function for half floats */
static inline half8 roundh8(half8 x, roundMode mode)
{
    // the compiler moves this condition outside the cycle automatically
    if (mode == roundMode::HALF_TO_EVEN)
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
    else
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

// @todo: investigate possibilities to get rid of float type.
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

static void bias_fp16_inner(const half8* __restrict__ in,
                            half8*       __restrict__ out,
                            const half8* __restrict__ /*weights*/,
                            const half8* __restrict__ biases,
                            int numLines,
                            int lineSize,
                            int /*baseLine*/,
                            const PostOpsNDParams* /*p*/)
{
    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int axis_i = 0;

    for (; axis_i < lineVectors - 3; axis_i += 4)
    {
        int i = 0;

        const half8* __restrict__ rin = in + axis_i;
        half8* __restrict__ rout = out + axis_i;

        half8 rb0 = biases[axis_i + 0];
        half8 rb1 = biases[axis_i + 1];
        half8 rb2 = biases[axis_i + 2];
        half8 rb3 = biases[axis_i + 3];

        if (numLines > 2)
        {
            half8 rr00 = rin[0] + rb0;
            half8 rr01 = rin[1] + rb1;
            half8 rr02 = rin[2] + rb2;
            half8 rr03 = rin[3] + rb3;
            rin += lineVectors;

            half8 rr10 = rin[0];
            half8 rr11 = rin[1];
            half8 rr12 = rin[2] + rb2;
            half8 rr13 = rin[3] + rb3;
            rin += lineVectors;

#pragma clang loop unroll_count(4)
            for (; i < numLines - 2; ++i)
            {
                half8 rr20 = rin[0];
                half8 rr21 = rin[1];
                half8 rr22 = rin[2];
                half8 rr23 = rin[3];
                rin += lineVectors;

                rout[0] = rr00;
                rout[1] = rr01;
                rout[2] = rr02;
                rout[3] = rr03;
                rout += lineVectors;

                rr00 = rr10 + rb0;
                rr01 = rr11 + rb1;
                rr02 = rr12;
                rr03 = rr13;

                rr10 = rr20;
                rr11 = rr21;
                rr12 = rr22 + rb2;
                rr13 = rr23 + rb3;
            }

            rin -= 2 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = rin[0] + rb0;
            rout[1] = rin[1] + rb1;
            rout[2] = rin[2] + rb2;
            rout[3] = rin[3] + rb3;
            rin += lineVectors;
            rout += lineVectors;
        }
    }

    for (; axis_i < lineVectors; ++axis_i)
    {
        int i = 0;

        const half8* __restrict__ rin = in + axis_i;
        half8* __restrict__ rout = out + axis_i;

        half8 rb0 = biases[axis_i];

        if (numLines > 2)
        {
            half8 rr0 = rin[0] + rb0;
            rin += lineVectors;

            half8 rr1 = rin[0];
            rin += lineVectors;

#pragma clang loop unroll_count(8)
            for (; i < numLines - 2; ++i)
            {
                half8 rr2 = rin[0];
                rin += lineVectors;

                rout[0] = rr0;
                rout += lineVectors;

                rr0 = rr1 + rb0;

                rr1 = rr2;
            }

            rin -= 2 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = rin[0] + rb0;
            rin  += lineVectors;
            rout += lineVectors;
        }
    }
}

static void bias_fp16_outer(const half8* __restrict__ in,
                            half8*       __restrict__ out,
                            const half8* __restrict__ /*weights*/,
                            const half8* __restrict__ _biases,
                            int numLines,
                            int lineSize,
                            int baseLine,
                            const PostOpsNDParams* p)
{
    const half* biases = (const half*)_biases;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < numLines - 7; line_i += 8)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;
        int axis_i1 = ((baseLine + line_i + 1) / p->axisGran) % p->axisDim;
        int axis_i2 = ((baseLine + line_i + 2) / p->axisGran) % p->axisDim;
        int axis_i3 = ((baseLine + line_i + 3) / p->axisGran) % p->axisDim;
        int axis_i4 = ((baseLine + line_i + 4) / p->axisGran) % p->axisDim;
        int axis_i5 = ((baseLine + line_i + 5) / p->axisGran) % p->axisDim;
        int axis_i6 = ((baseLine + line_i + 6) / p->axisGran) % p->axisDim;
        int axis_i7 = ((baseLine + line_i + 7) / p->axisGran) % p->axisDim;

        half8 rb0 = (half8) biases[axis_i0];
        half8 rb1 = (half8) biases[axis_i1];
        half8 rb2 = (half8) biases[axis_i2];
        half8 rb3 = (half8) biases[axis_i3];
        half8 rb4 = (half8) biases[axis_i4];
        half8 rb5 = (half8) biases[axis_i5];
        half8 rb6 = (half8) biases[axis_i6];
        half8 rb7 = (half8) biases[axis_i7];

        if (lineVectors >= 1)
        {
            int i = 0;

            half8 rr00 = in[(line_i + 0) * lineVectors + i] + rb0;
            half8 rr01 = in[(line_i + 1) * lineVectors + i] + rb1;
            half8 rr02 = in[(line_i + 2) * lineVectors + i] + rb2;
            half8 rr03 = in[(line_i + 3) * lineVectors + i] + rb3;
            half8 rr04 = in[(line_i + 4) * lineVectors + i];
            half8 rr05 = in[(line_i + 5) * lineVectors + i];
            half8 rr06 = in[(line_i + 6) * lineVectors + i];
            half8 rr07 = in[(line_i + 7) * lineVectors + i];

            for (i = 1; i < lineVectors; ++i)
            {
                half8 rr10 = in[(line_i + 0) * lineVectors + i] + rb0;
                half8 rr11 = in[(line_i + 1) * lineVectors + i] + rb1;
                half8 rr12 = in[(line_i + 2) * lineVectors + i] + rb2;
                half8 rr13 = in[(line_i + 3) * lineVectors + i] + rb3;
                half8 rr14 = in[(line_i + 4) * lineVectors + i];
                half8 rr15 = in[(line_i + 5) * lineVectors + i];
                half8 rr16 = in[(line_i + 6) * lineVectors + i];
                half8 rr17 = in[(line_i + 7) * lineVectors + i];

                out[(line_i + 0) * lineVectors + (i - 1)] = rr00;
                out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
                out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
                out[(line_i + 3) * lineVectors + (i - 1)] = rr03;
                out[(line_i + 4) * lineVectors + (i - 1)] = rr04 + rb4;
                out[(line_i + 5) * lineVectors + (i - 1)] = rr05 + rb5;
                out[(line_i + 6) * lineVectors + (i - 1)] = rr06 + rb6;
                out[(line_i + 7) * lineVectors + (i - 1)] = rr07 + rb7;

                rr00 = rr10;
                rr01 = rr11;
                rr02 = rr12;
                rr03 = rr13;
                rr04 = rr14;
                rr05 = rr15;
                rr06 = rr16;
                rr07 = rr17;
            }

            out[(line_i + 0) * lineVectors + (i - 1)] = rr00;
            out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
            out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
            out[(line_i + 3) * lineVectors + (i - 1)] = rr03;
            out[(line_i + 4) * lineVectors + (i - 1)] = rr04 + rb4;
            out[(line_i + 5) * lineVectors + (i - 1)] = rr05 + rb5;
            out[(line_i + 6) * lineVectors + (i - 1)] = rr06 + rb6;
            out[(line_i + 7) * lineVectors + (i - 1)] = rr07 + rb7;
        }
    }

    for (; line_i < numLines; ++line_i)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;

        half8 rb0 = (half8) biases[axis_i0];

        for (int i = 0; i < lineVectors; ++i)
        {
            half8 rr0 = in[(line_i + 0) * lineVectors + i] + rb0;

            out[(line_i + 0) * lineVectors + i] = rr0;
        }
    }
}

static void bias_leaky_relu_fp16_inner(const half8* __restrict__ in,
                                       half8*       __restrict__ out,
                                       const half8* __restrict__ /*weights*/,
                                       const half8* __restrict__ biases,
                                       int numLines,
                                       int lineSize,
                                       int /*baseLine*/,
                                       const PostOpsNDParams* p)
{
    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half8 zeros = (half8)0.0;
    const half8 xVal = (half8)x;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < lineVectors - 1; line_i += 2)
    {
        half8 rb0 = biases[line_i + 0];
        half8 rb1 = biases[line_i + 1];

        const half8* __restrict__ rin = in + line_i;
        half8* __restrict__ rout = out + line_i;

        int i = 0;

        if (numLines > 4)
        {
            half8 rr00 = rin[0] + rb0;
            half8 rr01 = rin[1] + rb1;
            rin += lineVectors;

            rr00 = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            rr01 = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);

            half8 rr10 = rin[0] + rb0;
            half8 rr11 = rin[1] + rb1;
            rin += lineVectors;

            half8 rq10 = __builtin_shave_cmu_min_f16_rr_half8(rr10, zeros); rr10 = __builtin_shave_cmu_max_f16_rr_half8(rr10, zeros);
            half8 rq11 = __builtin_shave_cmu_min_f16_rr_half8(rr11, zeros); rr11 = __builtin_shave_cmu_max_f16_rr_half8(rr11, zeros);

            half8 rr20 = rin[0] + rb0;
            half8 rr21 = rin[1] + rb1;
            rin += lineVectors;

            half8 rr30 = rin[0];
            half8 rr31 = rin[1];
            rin += lineVectors;

#pragma clang loop unroll_count(4)
            for (; i < numLines - 4; ++i)
            {
                half8 rr40 = rin[0];
                half8 rr41 = rin[1];
                rin += lineVectors;

                rout[0] = rr00;
                rout[1] = rr01;
                rout += lineVectors;

                rr00 = rr10 + xVal * rq10;
                rr01 = rr11 + xVal * rq11;

                rq10 = __builtin_shave_cmu_min_f16_rr_half8(rr20, zeros); rr10 = __builtin_shave_cmu_max_f16_rr_half8(rr20, zeros);
                rq11 = __builtin_shave_cmu_min_f16_rr_half8(rr21, zeros); rr11 = __builtin_shave_cmu_max_f16_rr_half8(rr21, zeros);

                rr20 = rr30 + rb0;
                rr21 = rr31 + rb1;

                rr30 = rr40;
                rr31 = rr41;
            }

            rin -= 4 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            half8 rr00 = rin[0] + rb0;
            half8 rr01 = rin[1] + rb1;
            rout[0] = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            rout[1] = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);
            rin += lineVectors;
            rout += lineVectors;
        }
    }

    for (; line_i < lineVectors; ++line_i)
    {
        half8 rb0 = biases[line_i];

        const half8* __restrict__ rin = in + line_i;
        half8* __restrict__ rout = out + line_i;

        int i = 0;

        if (numLines > 4)
        {
            half8 rr0 = rin[0] + rb0;
            rin += lineVectors;
            half8 rq0 = __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros); rr0 = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros);
            rr0 = rr0 + xVal * rq0;

            half8 rr1 = rin[0] + rb0;
            rin += lineVectors;
            half8 rq1 = __builtin_shave_cmu_min_f16_rr_half8(rr1, zeros); rr1 = __builtin_shave_cmu_max_f16_rr_half8(rr1, zeros);

            half8 rr2 = rin[0] + rb0;
            rin += lineVectors;

            half8 rr3 = rin[0];
            rin += lineVectors;

#pragma clang loop unroll_count(7)
            for (; i < numLines - 4; ++i)
            {
                half8 rr4 = rin[0];
                rin += lineVectors;

                rout[0] = rr0;
                rout += lineVectors;

                rr0 = rr1 + xVal * rq1;

                rq1 = __builtin_shave_cmu_min_f16_rr_half8(rr2, zeros); rr1 = __builtin_shave_cmu_max_f16_rr_half8(rr2, zeros);

                rr2 = rr3 + rb0;

                rr3 = rr4;
            }

            rin -= 4 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            half8 rr0 = rin[0] + rb0;
            rout[0] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros);
            rin += lineVectors;
            rout += lineVectors;
        }
    }
}

static void bias_leaky_relu_fp16_outer(const half8* __restrict__ in,
                                       half8*       __restrict__ out,
                                       const half8* __restrict__ /*weights*/,
                                       const half8* __restrict__ _biases,
                                       int numLines,
                                       int lineSize,
                                       int baseLine,
                                       const PostOpsNDParams* p)
{
    const half* biases = (const half*)_biases;

    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half8 zeros = (half8) half(0.0f);
    const half8 xVal = (half8)x;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < numLines - 7; line_i += 8)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;
        int axis_i1 = ((baseLine + line_i + 1) / p->axisGran) % p->axisDim;
        int axis_i2 = ((baseLine + line_i + 2) / p->axisGran) % p->axisDim;
        int axis_i3 = ((baseLine + line_i + 3) / p->axisGran) % p->axisDim;
        int axis_i4 = ((baseLine + line_i + 4) / p->axisGran) % p->axisDim;
        int axis_i5 = ((baseLine + line_i + 5) / p->axisGran) % p->axisDim;
        int axis_i6 = ((baseLine + line_i + 6) / p->axisGran) % p->axisDim;
        int axis_i7 = ((baseLine + line_i + 7) / p->axisGran) % p->axisDim;

        half8 rb0 = (half8) biases[axis_i0];
        half8 rb1 = (half8) biases[axis_i1];
        half8 rb2 = (half8) biases[axis_i2];
        half8 rb3 = (half8) biases[axis_i3];
        half8 rb4 = (half8) biases[axis_i4];
        half8 rb5 = (half8) biases[axis_i5];
        half8 rb6 = (half8) biases[axis_i6];
        half8 rb7 = (half8) biases[axis_i7];

        if (lineVectors >= 1)
        {
            int i = 0;

            half8 rr00 = in[(line_i + 0) * lineVectors + i] + rb0;
            half8 rr01 = in[(line_i + 1) * lineVectors + i] + rb1;
            half8 rr02 = in[(line_i + 2) * lineVectors + i] + rb2;
            half8 rr03 = in[(line_i + 3) * lineVectors + i] + rb3;
            half8 rr04 = in[(line_i + 4) * lineVectors + i] + rb4;
            half8 rr05 = in[(line_i + 5) * lineVectors + i] + rb5;
            half8 rr06 = in[(line_i + 6) * lineVectors + i] + rb6;
            half8 rr07 = in[(line_i + 7) * lineVectors + i] + rb7;

            half8 rq00 = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros);
            half8 rq01 = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros);
            half8 rq02 = __builtin_shave_cmu_max_f16_rr_half8(rr02, zeros);
            half8 rq03 = __builtin_shave_cmu_max_f16_rr_half8(rr03, zeros);
            half8 rq04 = __builtin_shave_cmu_max_f16_rr_half8(rr04, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr04, zeros);
            half8 rq05 = __builtin_shave_cmu_max_f16_rr_half8(rr05, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr05, zeros);
            half8 rq06 = __builtin_shave_cmu_max_f16_rr_half8(rr06, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr06, zeros);
            half8 rq07 = __builtin_shave_cmu_max_f16_rr_half8(rr07, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr07, zeros);

            for (i = 1; i < lineVectors; ++i)
            {
                half8 rr10 = in[(line_i + 0) * lineVectors + i] + rb0;
                half8 rr11 = in[(line_i + 1) * lineVectors + i] + rb1;
                half8 rr12 = in[(line_i + 2) * lineVectors + i] + rb2;
                half8 rr13 = in[(line_i + 3) * lineVectors + i] + rb3;
                half8 rr14 = in[(line_i + 4) * lineVectors + i] + rb4;
                half8 rr15 = in[(line_i + 5) * lineVectors + i] + rb5;
                half8 rr16 = in[(line_i + 6) * lineVectors + i] + rb6;
                half8 rr17 = in[(line_i + 7) * lineVectors + i] + rb7;

                half8 rq10 = __builtin_shave_cmu_max_f16_rr_half8(rr10, zeros);
                half8 rq11 = __builtin_shave_cmu_max_f16_rr_half8(rr11, zeros);
                half8 rq12 = __builtin_shave_cmu_max_f16_rr_half8(rr12, zeros);
                half8 rq13 = __builtin_shave_cmu_max_f16_rr_half8(rr13, zeros);
                half8 rq14 = __builtin_shave_cmu_max_f16_rr_half8(rr14, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr14, zeros);
                half8 rq15 = __builtin_shave_cmu_max_f16_rr_half8(rr15, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr15, zeros);
                half8 rq16 = __builtin_shave_cmu_max_f16_rr_half8(rr16, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr16, zeros);
                half8 rq17 = __builtin_shave_cmu_max_f16_rr_half8(rr17, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr17, zeros);

                out[(line_i + 0) * lineVectors + (i - 1)] = rq00 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
                out[(line_i + 1) * lineVectors + (i - 1)] = rq01 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);
                out[(line_i + 2) * lineVectors + (i - 1)] = rq02 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr02, zeros);
                out[(line_i + 3) * lineVectors + (i - 1)] = rq03 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr03, zeros);
                out[(line_i + 4) * lineVectors + (i - 1)] = rq04;
                out[(line_i + 5) * lineVectors + (i - 1)] = rq05;
                out[(line_i + 6) * lineVectors + (i - 1)] = rq06;
                out[(line_i + 7) * lineVectors + (i - 1)] = rq07;

                rq00 = rq10; rr00 = rr10;
                rq01 = rq11; rr01 = rr11;
                rq02 = rq12; rr02 = rr12;
                rq03 = rq13; rr03 = rr13;
                rq04 = rq14;
                rq05 = rq15;
                rq06 = rq16;
                rq07 = rq17;
            }

            out[(line_i + 0) * lineVectors + (i - 1)] = rq00 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            out[(line_i + 1) * lineVectors + (i - 1)] = rq01 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);
            out[(line_i + 2) * lineVectors + (i - 1)] = rq02 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr02, zeros);
            out[(line_i + 3) * lineVectors + (i - 1)] = rq03 + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr03, zeros);
            out[(line_i + 4) * lineVectors + (i - 1)] = rq04;
            out[(line_i + 5) * lineVectors + (i - 1)] = rq05;
            out[(line_i + 6) * lineVectors + (i - 1)] = rq06;
            out[(line_i + 7) * lineVectors + (i - 1)] = rq07;
        }
    }

    for (; line_i < numLines; ++line_i)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;

        half8 rb0 = (half8) biases[axis_i0];

        for (int i = 0; i < lineVectors; ++i)
        {
            half8 rr0 = in[(line_i + 0) * lineVectors + i] + rb0;

            out[(line_i + 0) * lineVectors + i] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros);
        }
    }
}

static void bias_relu_fp16_inner(const half8* __restrict__ in,
                                 half8*       __restrict__ out,
                                 const half8* __restrict__ /*weights*/,
                                 const half8* __restrict__ biases,
                                 int numLines,
                                 int lineSize,
                                 int /*baseLine*/,
                                 const PostOpsNDParams* p)
{
    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half8 xVal = (x <= half(0.0)) ? (half8)X_MAX : (half8)x;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < lineVectors - 3; line_i += 4)
    {
        half8 rb0 = biases[line_i + 0];
        half8 rb1 = biases[line_i + 1];
        half8 rb2 = biases[line_i + 2];
        half8 rb3 = biases[line_i + 3];

        const half8* __restrict__ rin = in + line_i;
        half8* __restrict__ rout = out + line_i;

        int i = 0;

        if (numLines > 3)
        {
            half8 rr00 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[0] + rb0, xVal);
            half8 rr01 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[1] + rb1, xVal);
            half8 rr02 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[2] + rb2, xVal);
            half8 rr03 = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[3] + rb3, xVal);
            rin += lineVectors;

            half8 rr10 = rin[0] + rb0;
            half8 rr11 = rin[1] + rb1;
            half8 rr12 = rin[2] + rb2;
            half8 rr13 = rin[3] + rb3;
            rin += lineVectors;

            half8 rr20 = rin[0];
            half8 rr21 = rin[1];
            half8 rr22 = rin[2];
            half8 rr23 = rin[3];
            rin += lineVectors;

#pragma clang loop unroll_count(4)
            for (; i < numLines - 3; ++i)
            {
                half8 rr30 = rin[0];
                half8 rr31 = rin[1];
                half8 rr32 = rin[2];
                half8 rr33 = rin[3];
                rin += lineVectors;

                rout[0] = rr00;
                rout[1] = rr01;
                rout[2] = rr02;
                rout[3] = rr03;
                rout += lineVectors;

                rr00 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr10, xVal);
                rr01 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr11, xVal);
                rr02 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr12, xVal);
                rr03 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr13, xVal);

                rr10 = rr20 + rb0;
                rr11 = rr21 + rb1;
                rr12 = rr22 + rb2;
                rr13 = rr23 + rb3;

                rr20 = rr30;
                rr21 = rr31;
                rr22 = rr32;
                rr23 = rr33;
            }

            rin -= 3 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[0] + rb0, xVal);
            rout[1] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[1] + rb1, xVal);
            rout[2] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[2] + rb2, xVal);
            rout[3] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[3] + rb3, xVal);
            rin += lineVectors;
            rout += lineVectors;
        }
    }

    for (; line_i < lineVectors; ++line_i)
    {
        half8 rb0 = biases[line_i];

        const half8* __restrict__ rin = in + line_i;
        half8* __restrict__ rout = out + line_i;

        int i = 0;

        if (numLines > 3)
        {
            half8 rr0 = rin[0];
            rin += lineVectors;

            rr0 = rr0 + rb0;
            rr0 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr0, xVal);

            half8 rr1 = rin[0];
            rin += lineVectors;

            rr1 = rr1 + rb0;

            half8 rr2 = rin[0];
            rin += lineVectors;

#pragma clang loop unroll_count(8)
            for (; i < numLines - 3; ++i)
            {
                half8 rr3 = rin[0];
                rin += lineVectors;

                rout[0] = rr0;
                rout += lineVectors;

                rr0 = __builtin_shave_cmu_clamp0_f16_rr_half8(rr1, xVal);

                rr1 = rr2 + rb0;

                rr2 = rr3;
            }

            rin -= 3 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = __builtin_shave_cmu_clamp0_f16_rr_half8(rin[0] + rb0, xVal);
            rin += lineVectors;
            rout += lineVectors;
        }
    }
}

static void bias_relu_fp16_outer(const half8* __restrict__ in,
                                 half8*       __restrict__ out,
                                 const half8* __restrict__ /*weights*/,
                                 const half8* __restrict__ _biases,
                                 int numLines,
                                 int lineSize,
                                 int baseLine,
                                 const PostOpsNDParams* p)
{
    const half* biases = (const half*)_biases;

    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half8 xVal = (x <= half(0.0)) ? (half8)X_MAX : (half8)x;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < numLines - 7; line_i += 8)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;
        int axis_i1 = ((baseLine + line_i + 1) / p->axisGran) % p->axisDim;
        int axis_i2 = ((baseLine + line_i + 2) / p->axisGran) % p->axisDim;
        int axis_i3 = ((baseLine + line_i + 3) / p->axisGran) % p->axisDim;
        int axis_i4 = ((baseLine + line_i + 4) / p->axisGran) % p->axisDim;
        int axis_i5 = ((baseLine + line_i + 5) / p->axisGran) % p->axisDim;
        int axis_i6 = ((baseLine + line_i + 6) / p->axisGran) % p->axisDim;
        int axis_i7 = ((baseLine + line_i + 7) / p->axisGran) % p->axisDim;

        half8 rb0 = (half8) biases[axis_i0];
        half8 rb1 = (half8) biases[axis_i1];
        half8 rb2 = (half8) biases[axis_i2];
        half8 rb3 = (half8) biases[axis_i3];
        half8 rb4 = (half8) biases[axis_i4];
        half8 rb5 = (half8) biases[axis_i5];
        half8 rb6 = (half8) biases[axis_i6];
        half8 rb7 = (half8) biases[axis_i7];

        if (lineVectors >= 1)
        {
            int i = 0;

            half8 rr00 = __builtin_shave_cmu_clamp0_f16_rr_half8(in[(line_i + 0) * lineVectors + i] + rb0, xVal);
            half8 rr01 = __builtin_shave_cmu_clamp0_f16_rr_half8(in[(line_i + 1) * lineVectors + i] + rb1, xVal);
            half8 rr02 = __builtin_shave_cmu_clamp0_f16_rr_half8(in[(line_i + 2) * lineVectors + i] + rb2, xVal);
            half8 rr03 = in[(line_i + 3) * lineVectors + i];
            half8 rr04 = in[(line_i + 4) * lineVectors + i];
            half8 rr05 = in[(line_i + 5) * lineVectors + i];
            half8 rr06 = in[(line_i + 6) * lineVectors + i];
            half8 rr07 = in[(line_i + 7) * lineVectors + i];

            for (i = 1; i < lineVectors; ++i)
            {
                half8 rr10 = __builtin_shave_cmu_clamp0_f16_rr_half8(in[(line_i + 0) * lineVectors + i] + rb0, xVal);
                half8 rr11 = __builtin_shave_cmu_clamp0_f16_rr_half8(in[(line_i + 1) * lineVectors + i] + rb1, xVal);
                half8 rr12 = __builtin_shave_cmu_clamp0_f16_rr_half8(in[(line_i + 2) * lineVectors + i] + rb2, xVal);
                half8 rr13 = in[(line_i + 3) * lineVectors + i];
                half8 rr14 = in[(line_i + 4) * lineVectors + i];
                half8 rr15 = in[(line_i + 5) * lineVectors + i];
                half8 rr16 = in[(line_i + 6) * lineVectors + i];
                half8 rr17 = in[(line_i + 7) * lineVectors + i];

                out[(line_i + 0) * lineVectors + (i - 1)] = rr00;
                out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
                out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
                out[(line_i + 3) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr03 + rb3, xVal);
                out[(line_i + 4) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr04 + rb4, xVal);
                out[(line_i + 5) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr05 + rb5, xVal);
                out[(line_i + 6) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr06 + rb6, xVal);
                out[(line_i + 7) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr07 + rb7, xVal);

                rr00 = rr10;
                rr01 = rr11;
                rr02 = rr12;
                rr03 = rr13;
                rr04 = rr14;
                rr05 = rr15;
                rr06 = rr16;
                rr07 = rr17;
            }

            out[(line_i + 0) * lineVectors + (i - 1)] = rr00;
            out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
            out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
            out[(line_i + 3) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr03 + rb3, xVal);
            out[(line_i + 4) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr04 + rb4, xVal);
            out[(line_i + 5) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr05 + rb5, xVal);
            out[(line_i + 6) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr06 + rb6, xVal);
            out[(line_i + 7) * lineVectors + (i - 1)] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr07 + rb7, xVal);
        }
    }

    for (; line_i < numLines; ++line_i)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;

        half8 rb0 = (half8) biases[axis_i0];

        for (int i = 0; i < lineVectors; ++i)
        {
            half8 rr0 = in[(line_i + 0) * lineVectors + i] + rb0;

            out[(line_i + 0) * lineVectors + i] = __builtin_shave_cmu_clamp0_f16_rr_half8(rr0, xVal);
        }
    }
}

static void clamp_fp16(const half8* __restrict__ in,
                       half8*       __restrict__ out,
                       const half8* __restrict__ /*weights*/,
                       const half8* __restrict__ /*biases*/,
                       int numLines,
                       int lineSize,
                       int /*baseLine*/,
                       const PostOpsNDParams* p)
{
    const t_ClampLayerParams* clampParams = reinterpret_cast<const t_ClampLayerParams*>(p->params);

    const half8 minVal = (half8) (clampParams->min);
    const half8 maxVal = (half8) (clampParams->max);

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    int i = 0;

    if (numVectors >= 8)
    {
        half8 r00 = in[0];
        half8 r01 = in[1];
        half8 r02 = in[2];
        half8 r03 = in[3];
        half8 r04 = in[4];
        half8 r05 = in[5];
        half8 r06 = in[6];
        half8 r07 = in[7];
        in += 8;

        for (i = 8; i < numVectors - 7; i += 8)
        {
            half8 r10 = in[0];
            half8 r11 = in[1];
            half8 r12 = in[2];
            half8 r13 = in[3];
            half8 r14 = in[4];
            half8 r15 = in[5];
            half8 r16 = in[6];
            half8 r17 = in[7];
            in += 8;

            out[0] = __builtin_shave_cmu_clampab_f16_rrr_half8(r00, minVal, maxVal);
            out[1] = __builtin_shave_cmu_clampab_f16_rrr_half8(r01, minVal, maxVal);
            out[2] = __builtin_shave_cmu_clampab_f16_rrr_half8(r02, minVal, maxVal);
            out[3] = __builtin_shave_cmu_clampab_f16_rrr_half8(r03, minVal, maxVal);
            out[4] = __builtin_shave_cmu_clampab_f16_rrr_half8(r04, minVal, maxVal);
            out[5] = __builtin_shave_cmu_clampab_f16_rrr_half8(r05, minVal, maxVal);
            out[6] = __builtin_shave_cmu_clampab_f16_rrr_half8(r06, minVal, maxVal);
            out[7] = __builtin_shave_cmu_clampab_f16_rrr_half8(r07, minVal, maxVal);
            out += 8;

            r00 = r10;
            r01 = r11;
            r02 = r12;
            r03 = r13;
            r04 = r14;
            r05 = r15;
            r06 = r16;
            r07 = r17;
        }

        out[0] = __builtin_shave_cmu_clampab_f16_rrr_half8(r00, minVal, maxVal);
        out[1] = __builtin_shave_cmu_clampab_f16_rrr_half8(r01, minVal, maxVal);
        out[2] = __builtin_shave_cmu_clampab_f16_rrr_half8(r02, minVal, maxVal);
        out[3] = __builtin_shave_cmu_clampab_f16_rrr_half8(r03, minVal, maxVal);
        out[4] = __builtin_shave_cmu_clampab_f16_rrr_half8(r04, minVal, maxVal);
        out[5] = __builtin_shave_cmu_clampab_f16_rrr_half8(r05, minVal, maxVal);
        out[6] = __builtin_shave_cmu_clampab_f16_rrr_half8(r06, minVal, maxVal);
        out[7] = __builtin_shave_cmu_clampab_f16_rrr_half8(r07, minVal, maxVal);
        out += 8;
    }

    for (; i < numVectors; ++i)
        *out++ = __builtin_shave_cmu_clampab_f16_rrr_half8(*in++, minVal, maxVal);
}

static void elu_fp16(const half8* __restrict__ in,
                     half8*       __restrict__ out,
                     const half8* __restrict__ /*weights*/,
                     const half8* __restrict__ /*biases*/,
                     int numLines,
                     int lineSize,
                     int /*baseLine*/,
                     const PostOpsNDParams* p)
{
    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half alpha = x;
    const half8 zero = (half8)0.0f;

    const uint16_t inv_ln2 = 0x3dc6;
    const half inv_ln2_h = *(const half*)&inv_ln2;
    const half8 vinv_ln2 = (half8)inv_ln2_h;

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

#pragma clang loop unroll_count(8)
    for (int i = 0; i < numVectors; ++i)
    {
        half8 min = __builtin_shave_cmu_min_f16_rr_half8(in[i], zero);
        half8 max = __builtin_shave_cmu_max_f16_rr_half8(in[i], zero) - alpha;

        half8 exp_x = min * vinv_ln2;
        exp2_vec(exp_x, exp_x);

        out[i] = max + alpha * exp_x;
    }
}

static void leaky_relu_fp16(const half8* __restrict__ in,
                            half8*       __restrict__ out,
                            const half8* __restrict__ /*weights*/,
                            const half8* __restrict__ /*biases*/,
                            int numLines,
                            int lineSize,
                            int /*baseLine*/,
                            const PostOpsNDParams* p)
{
    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half8 zeros = (half8)0.0;
    const half8 xVal = (half8)x;

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    int i = 0;

    if (numVectors >= 8)
    {
        half8 r00 = in[0];
        half8 r01 = in[1];
        half8 r02 = in[2];
        half8 r03 = in[3];
        half8 r04 = in[4];
        half8 r05 = in[5];
        half8 r06 = in[6];
        half8 r07 = in[7];
        in += 8;

        half8 q00 = __builtin_shave_cmu_min_f16_rr_half8(r00, zeros);
        half8 q01 = __builtin_shave_cmu_min_f16_rr_half8(r01, zeros);
        half8 q02 = __builtin_shave_cmu_min_f16_rr_half8(r02, zeros);
        half8 q03 = __builtin_shave_cmu_min_f16_rr_half8(r03, zeros);
        half8 q04 = __builtin_shave_cmu_min_f16_rr_half8(r04, zeros);
        half8 q05 = __builtin_shave_cmu_min_f16_rr_half8(r05, zeros);
        half8 q06 = __builtin_shave_cmu_max_f16_rr_half8(r06, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(r06, zeros);
        half8 q07 = __builtin_shave_cmu_max_f16_rr_half8(r07, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(r07, zeros);

        for (i = 8; i < numVectors - 7; i += 8)
        {
            half8 r10 = in[0];
            half8 r11 = in[1];
            half8 r12 = in[2];
            half8 r13 = in[3];
            half8 r14 = in[4];
            half8 r15 = in[5];
            half8 r16 = in[6];
            half8 r17 = in[7];
            in += 8;

            half8 q10 = __builtin_shave_cmu_min_f16_rr_half8(r10, zeros);
            half8 q11 = __builtin_shave_cmu_min_f16_rr_half8(r11, zeros);
            half8 q12 = __builtin_shave_cmu_min_f16_rr_half8(r12, zeros);
            half8 q13 = __builtin_shave_cmu_min_f16_rr_half8(r13, zeros);
            half8 q14 = __builtin_shave_cmu_min_f16_rr_half8(r14, zeros);
            half8 q15 = __builtin_shave_cmu_min_f16_rr_half8(r15, zeros);
            half8 q16 = __builtin_shave_cmu_max_f16_rr_half8(r16, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(r16, zeros);
            half8 q17 = __builtin_shave_cmu_max_f16_rr_half8(r17, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(r17, zeros);

            out[0] = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + xVal * q00;
            out[1] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + xVal * q01;
            out[2] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + xVal * q02;
            out[3] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + xVal * q03;
            out[4] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + xVal * q04;
            out[5] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + xVal * q05;
            out[6] = q06;
            out[7] = q07;
            out += 8;

            r00 = r10; q00 = q10;
            r01 = r11; q01 = q11;
            r02 = r12; q02 = q12;
            r03 = r13; q03 = q13;
            r04 = r14; q04 = q14;
            r05 = r15; q05 = q15;
            r06 = r16; q06 = q16;
            r07 = r17; q07 = q17;
        }

        out[0] = __builtin_shave_cmu_max_f16_rr_half8(r00, zeros) + xVal * q00;
        out[1] = __builtin_shave_cmu_max_f16_rr_half8(r01, zeros) + xVal * q01;
        out[2] = __builtin_shave_cmu_max_f16_rr_half8(r02, zeros) + xVal * q02;
        out[3] = __builtin_shave_cmu_max_f16_rr_half8(r03, zeros) + xVal * q03;
        out[4] = __builtin_shave_cmu_max_f16_rr_half8(r04, zeros) + xVal * q04;
        out[5] = __builtin_shave_cmu_max_f16_rr_half8(r05, zeros) + xVal * q05;
        out[6] = q06;
        out[7] = q07;
        out += 8;
    }

    for (; i < numVectors; ++i)
    {
        half8 r = *in++;
        *out++ = __builtin_shave_cmu_max_f16_rr_half8(r, zeros) + xVal * __builtin_shave_cmu_min_f16_rr_half8(r, zeros);
    }
}

static void power_fp16(const half8* __restrict__ in,
                       half8*       __restrict__ out,
                       const half8* __restrict__ /*weights*/,
                       const half8* __restrict__ /*biases*/,
                       int numLines,
                       int lineSize,
                       int /*baseLine*/,
                       const PostOpsNDParams* p)
{
    const t_PowerLayerParams* powerParams = reinterpret_cast<const t_PowerLayerParams*>(p->params);

    const half8 shift = (half8) powerParams->shift;
    const float shift_scal = powerParams->shift;

    const half8 scale = (half8) powerParams->scale;
    const float scale_scal = powerParams->scale;

    const half8 power = (half8) powerParams->power;
    const float power_scal = powerParams->power;

    // Compute power(x) = (shift + scale * x)^power
    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    if ((scale_scal == 1.0f) && (power_scal == 1.0f) && (shift_scal == 0.0f)) // out = in (copying)
    {
        // Do not do anything; src is already copied to dst CMX tile
        return;
    }

    if (power_scal == 0.0f) // power == 0
    {
        half8 vfill = (half8)1.0f;
#pragma clang loop unroll_count(8)
        for (int i = 0; i < numVectors; ++i)
        {
            out[i] = vfill;
        }
    }
    else
    {
        bool is_power_integer = floorf(fabsf(power_scal)) == fabsf(power_scal);
        const int integer_power = fabsf(power_scal);

        if (is_power_integer) // power is integer
        {
            if (integer_power == 1) // power == 1
            {
                if ((scale_scal == -1.0f) && (shift_scal == 0.0f)) // out = -in
                {
                    half8 vec;
                    if (numVectors)
                    {
                        vec = -in[0];
                    }
#pragma clang loop unroll_count(4)
                    for (int i = 1; i < numVectors; ++i)
                    {
                        out[i-1] = vec;
                        vec = -in[i];
                    }
                    if (numVectors)
                    {
                        out[numVectors-1] = vec;
                    }
                }
                else
                {
                    half8 v;

                    if (numVectors)
                    {
                        v = scale * in[0];
                    }
#pragma clang loop unroll_count(8)
                    for (int i = 1; i < numVectors; ++i)
                    {
                        half8 vres = v + shift;
                        v = scale * in[i];
                        out[i-1] = vres;
                    }
                    if (numVectors)
                    {
                        out[numVectors-1] = v + shift;
                    }
                }

            }
            else if (integer_power == 2) // power == 2
            {
                half8 base;

                if (numVectors)
                {
                    base = shift + scale * in[0];
                }
#pragma clang loop unroll_count(8)
                for (int i = 1; i < numVectors; ++i)
                {
                    half8 vres = (base) * (base);
                    base = shift + scale * in[i];
                    out[i-1] = vres;
                }
                if (numVectors)
                {
                    out[numVectors-1] = (base)*(base);
                }
            }
            else if (integer_power == 3) // power == 3
            {
                half8 vin;
                half8 base;
                half8 vres;

                // 0 iteration
                if (numVectors)
                {
                    half8 vin = in[0];
                    base = shift + scale * vin;
                    vres = base * base * base;
                }
                // 1 iteration
                if (numVectors > 1)
                {
                    vin = in[1];
                    base = shift + scale * vin;
                }
#pragma clang loop unroll_count(8)
                for (int i = 2; i < numVectors; ++i)
                {
                    out[i-2] = vres;
                    vres = base * base * base;
                    half8 vin = in[i];
                    base = shift + scale * vin;
                }
                // (numVectors-2) iteration
                if (numVectors > 1)
                {
                    out[numVectors-2] = vres;
                }
                // (numVectors-1) iteration
                if (numVectors > 0)
                {
                    out[numVectors-1] = (base)*(base)*(base);
                }
            }
            else // general integer power
            {
                int i = 0;
#pragma clang loop unroll_count(1)
                for (; i < numVectors-7; i+=8)
                {
                    half8 base0 = shift + scale * in[i];
                    half8 base1 = shift + scale * in[i+1];
                    half8 base2 = shift + scale * in[i+2];
                    half8 base3 = shift + scale * in[i+3];
                    half8 base4 = shift + scale * in[i+4];
                    half8 base5 = shift + scale * in[i+5];
                    half8 base6 = shift + scale * in[i+6];
                    half8 base7 = shift + scale * in[i+7];

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
                    out[i+0] = res0;
                    out[i+1] = res1;
                    out[i+2] = res2;
                    out[i+3] = res3;
                    out[i+4] = res4;
                    out[i+5] = res5;
                    out[i+6] = res6;
                    out[i+7] = res7;
                }

                for (; i < numVectors; ++i)
                {
                    half8 base = shift + scale * in[i];

                    half8 res = base;
                    for (int p = 0; p < integer_power-1; p++)
                    {
                        res = res * base;
                    }
                    out[i] = res;
                }
            }

            if (power_scal < 0.0f)
            {
                half8 v;
                if (numVectors)
                {
                    v = in[0];
                }
#pragma clang loop unroll_count(8)
                for (int i = 1; i < numVectors; ++i)
                {
                    out[i-1] = 1.f / v;
                    v = in[i];
                }
                if (numVectors)
                {
                    out[numVectors-1] = 1.f / v;
                }
            }
        }
        else // general case
        {
            half8 base_log = (half8)0, base_mult = (half8)0, base = (half8)0;

            // 0 iteration
            if (numVectors > 0)
            {
                base = shift + scale * in[0];
                log2_vec(base, base_log);
            }
            // 1 iteration
            if (numVectors > 1)
            {
                base = shift + scale * in[1];
            }

#pragma clang loop unroll_count(8)
            for (int i = 2; i < numVectors; ++i)
            {
                base_mult = base_log * power;// 2 stage
                exp2_vec(base_mult, out[i-2]);

                log2_vec(base, base_log);// 1 stage

                base = shift + scale * in[i];// 0 stage
            }

            // (numVectors-2) iteration
            if (numVectors > 1)
            {
                base_mult = base_log * power;
                exp2_vec(base_mult, out[numVectors-2]);
            }
            // (numVectors-1) iteration
            if (numVectors > 0)
            {
                log2_vec(base, base_log);
                base_mult = base_log * power;
                exp2_vec(base_mult, out[numVectors-1]);
            }
        }
    }
}

static void prelu_fp16_inner(const half8* __restrict__ in,
                             half8*       __restrict__ out,
                             const half8* __restrict__ weights,
                             const half8* __restrict__ /*biases*/,
                             int numLines,
                             int lineSize,
                             int /*baseLine*/,
                             const PostOpsNDParams* /*p*/)
{
    const half8 zeros = (half8)0.0;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    for (int line_i = 0; line_i < lineVectors; ++line_i)
    {
        half8 rw0 = weights[line_i];

        const half8* __restrict__ rin = in + line_i;
        half8* __restrict__ rout = out + line_i;

        int i = 0;

        if (numLines >= 8)
        {
            half8 rr00 = rin[0 * lineVectors];
            half8 rr01 = rin[1 * lineVectors];
            half8 rr02 = rin[2 * lineVectors];
            half8 rr03 = rin[3 * lineVectors];
            half8 rr04 = rin[4 * lineVectors];
            half8 rr05 = rin[5 * lineVectors];
            half8 rr06 = rin[6 * lineVectors];
            half8 rr07 = rin[7 * lineVectors];
            rin += 8 * lineVectors;

            half8 rq00 = __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            half8 rq01 = __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);
            half8 rq02 = __builtin_shave_cmu_min_f16_rr_half8(rr02, zeros);
            half8 rq03 = __builtin_shave_cmu_min_f16_rr_half8(rr03, zeros);

            for (i = 8; i < numLines - 7; i += 8)
            {
                half8 rr10 = rin[0 * lineVectors];
                half8 rr11 = rin[1 * lineVectors];
                half8 rr12 = rin[2 * lineVectors];
                half8 rr13 = rin[3 * lineVectors];
                half8 rr14 = rin[4 * lineVectors];
                half8 rr15 = rin[5 * lineVectors];
                half8 rr16 = rin[6 * lineVectors];
                half8 rr17 = rin[7 * lineVectors];
                rin += 8 * lineVectors;

                half8 q10 = __builtin_shave_cmu_min_f16_rr_half8(rr10, zeros);
                half8 q11 = __builtin_shave_cmu_min_f16_rr_half8(rr11, zeros);
                half8 q12 = __builtin_shave_cmu_min_f16_rr_half8(rr12, zeros);
                half8 q13 = __builtin_shave_cmu_min_f16_rr_half8(rr13, zeros);

                rout[0 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + rw0 * rq00;
                rout[1 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + rw0 * rq01;
                rout[2 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr02, zeros) + rw0 * rq02;
                rout[3 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr03, zeros) + rw0 * rq03;
                rout[4 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr04, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr04, zeros);
                rout[5 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr05, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr05, zeros);
                rout[6 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr06, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr06, zeros);
                rout[7 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr07, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr07, zeros);
                rout += 8 * lineVectors;

                rr00 = rr10; rq00 = q10;
                rr01 = rr11; rq01 = q11;
                rr02 = rr12; rq02 = q12;
                rr03 = rr13; rq03 = q13;
                rr04 = rr14;
                rr05 = rr15;
                rr06 = rr16;
                rr07 = rr17;
            }

            rout[0 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + rw0 * rq00;
            rout[1 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + rw0 * rq01;
            rout[2 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr02, zeros) + rw0 * rq02;
            rout[3 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr03, zeros) + rw0 * rq03;
            rout[4 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr04, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr04, zeros);
            rout[5 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr05, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr05, zeros);
            rout[6 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr06, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr06, zeros);
            rout[7 * lineVectors] = __builtin_shave_cmu_max_f16_rr_half8(rr07, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr07, zeros);
            rout += 8 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            half8 r = rin[0];
            rout[0] = __builtin_shave_cmu_max_f16_rr_half8(r, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(r, zeros);
            rin += lineVectors;
            rout += lineVectors;
        }
    }
}

static void prelu_fp16_outer(const half8* __restrict__ in,
                             half8*       __restrict__ out,
                             const half8* __restrict__ _weights,
                             const half8* __restrict__ /*biases*/,
                             int numLines,
                             int lineSize,
                             int baseLine,
                             const PostOpsNDParams* p)
{
    const half* weights = (const half*)_weights;

    const half8 zeros = (half8) half(0.0f);

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < numLines - 7; line_i += 8)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;
        int axis_i1 = ((baseLine + line_i + 1) / p->axisGran) % p->axisDim;
        int axis_i2 = ((baseLine + line_i + 2) / p->axisGran) % p->axisDim;
        int axis_i3 = ((baseLine + line_i + 3) / p->axisGran) % p->axisDim;
        int axis_i4 = ((baseLine + line_i + 4) / p->axisGran) % p->axisDim;
        int axis_i5 = ((baseLine + line_i + 5) / p->axisGran) % p->axisDim;
        int axis_i6 = ((baseLine + line_i + 6) / p->axisGran) % p->axisDim;
        int axis_i7 = ((baseLine + line_i + 7) / p->axisGran) % p->axisDim;

        half8 rw0 = (half8) weights[axis_i0];
        half8 rw1 = (half8) weights[axis_i1];
        half8 rw2 = (half8) weights[axis_i2];
        half8 rw3 = (half8) weights[axis_i3];
        half8 rw4 = (half8) weights[axis_i4];
        half8 rw5 = (half8) weights[axis_i5];
        half8 rw6 = (half8) weights[axis_i6];
        half8 rw7 = (half8) weights[axis_i7];

        if (lineVectors >= 1)
        {
            int i = 0;

            half8 rr00 = in[(line_i + 0) * lineVectors + i];
            half8 rr01 = in[(line_i + 1) * lineVectors + i];
            half8 rr02 = in[(line_i + 2) * lineVectors + i];
            half8 rr03 = in[(line_i + 3) * lineVectors + i];
            half8 rr04 = in[(line_i + 4) * lineVectors + i];
            half8 rr05 = in[(line_i + 5) * lineVectors + i];
            half8 rr06 = in[(line_i + 6) * lineVectors + i];
            half8 rr07 = in[(line_i + 7) * lineVectors + i];

            half8 rq00 = __builtin_shave_cmu_max_f16_rr_half8(rr00, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr00, zeros);
            half8 rq01 = __builtin_shave_cmu_max_f16_rr_half8(rr01, zeros) + rw1 * __builtin_shave_cmu_min_f16_rr_half8(rr01, zeros);
            half8 rq02 = __builtin_shave_cmu_max_f16_rr_half8(rr02, zeros);
            half8 rq03 = __builtin_shave_cmu_max_f16_rr_half8(rr03, zeros);
            half8 rq04 = __builtin_shave_cmu_max_f16_rr_half8(rr04, zeros);
            half8 rq05 = __builtin_shave_cmu_max_f16_rr_half8(rr05, zeros);
            half8 rq06 = __builtin_shave_cmu_max_f16_rr_half8(rr06, zeros);
            half8 rq07 = __builtin_shave_cmu_max_f16_rr_half8(rr07, zeros);

            for (i = 1; i < lineVectors; ++i)
            {
                half8 rr10 = in[(line_i + 0) * lineVectors + i];
                half8 rr11 = in[(line_i + 1) * lineVectors + i];
                half8 rr12 = in[(line_i + 2) * lineVectors + i];
                half8 rr13 = in[(line_i + 3) * lineVectors + i];
                half8 rr14 = in[(line_i + 4) * lineVectors + i];
                half8 rr15 = in[(line_i + 5) * lineVectors + i];
                half8 rr16 = in[(line_i + 6) * lineVectors + i];
                half8 rr17 = in[(line_i + 7) * lineVectors + i];

                half8 rq10 = __builtin_shave_cmu_max_f16_rr_half8(rr10, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr10, zeros);
                half8 rq11 = __builtin_shave_cmu_max_f16_rr_half8(rr11, zeros) + rw1 * __builtin_shave_cmu_min_f16_rr_half8(rr11, zeros);
                half8 rq12 = __builtin_shave_cmu_max_f16_rr_half8(rr12, zeros);
                half8 rq13 = __builtin_shave_cmu_max_f16_rr_half8(rr13, zeros);
                half8 rq14 = __builtin_shave_cmu_max_f16_rr_half8(rr14, zeros);
                half8 rq15 = __builtin_shave_cmu_max_f16_rr_half8(rr15, zeros);
                half8 rq16 = __builtin_shave_cmu_max_f16_rr_half8(rr16, zeros);
                half8 rq17 = __builtin_shave_cmu_max_f16_rr_half8(rr17, zeros);

                out[(line_i + 0) * lineVectors + (i - 1)] = rq00;
                out[(line_i + 1) * lineVectors + (i - 1)] = rq01;
                out[(line_i + 2) * lineVectors + (i - 1)] = rq02 + rw2 * __builtin_shave_cmu_min_f16_rr_half8(rr02, zeros);
                out[(line_i + 3) * lineVectors + (i - 1)] = rq03 + rw3 * __builtin_shave_cmu_min_f16_rr_half8(rr03, zeros);
                out[(line_i + 4) * lineVectors + (i - 1)] = rq04 + rw4 * __builtin_shave_cmu_min_f16_rr_half8(rr04, zeros);
                out[(line_i + 5) * lineVectors + (i - 1)] = rq05 + rw5 * __builtin_shave_cmu_min_f16_rr_half8(rr05, zeros);
                out[(line_i + 6) * lineVectors + (i - 1)] = rq06 + rw6 * __builtin_shave_cmu_min_f16_rr_half8(rr06, zeros);
                out[(line_i + 7) * lineVectors + (i - 1)] = rq07 + rw7 * __builtin_shave_cmu_min_f16_rr_half8(rr07, zeros);

                rq00 = rq10;
                rq01 = rq11;
                rq02 = rq12; rr02 = rr12;
                rq03 = rq13; rr03 = rr13;
                rq04 = rq14; rr04 = rr14;
                rq05 = rq15; rr05 = rr15;
                rq06 = rq16; rr06 = rr16;
                rq07 = rq17; rr07 = rr17;
            }

            out[(line_i + 0) * lineVectors + (i - 1)] = rq00;
            out[(line_i + 1) * lineVectors + (i - 1)] = rq01;
            out[(line_i + 2) * lineVectors + (i - 1)] = rq02 + rw2 * __builtin_shave_cmu_min_f16_rr_half8(rr02, zeros);
            out[(line_i + 3) * lineVectors + (i - 1)] = rq03 + rw3 * __builtin_shave_cmu_min_f16_rr_half8(rr03, zeros);
            out[(line_i + 4) * lineVectors + (i - 1)] = rq04 + rw4 * __builtin_shave_cmu_min_f16_rr_half8(rr04, zeros);
            out[(line_i + 5) * lineVectors + (i - 1)] = rq05 + rw5 * __builtin_shave_cmu_min_f16_rr_half8(rr05, zeros);
            out[(line_i + 6) * lineVectors + (i - 1)] = rq06 + rw6 * __builtin_shave_cmu_min_f16_rr_half8(rr06, zeros);
            out[(line_i + 7) * lineVectors + (i - 1)] = rq07 + rw7 * __builtin_shave_cmu_min_f16_rr_half8(rr07, zeros);
        }
    }

    for (; line_i < numLines; ++line_i)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;

        half8 rw0 = (half8) weights[axis_i0];

        for (int i = 0; i < lineVectors; ++i)
        {
            half8 rr0 = in[(line_i + 0) * lineVectors + i];

            out[(line_i + 0) * lineVectors + i] = __builtin_shave_cmu_max_f16_rr_half8(rr0, zeros) + rw0 * __builtin_shave_cmu_min_f16_rr_half8(rr0, zeros);
        }
    }
}

static void relu_fp16(const half8* __restrict__ in,
                      half8*       __restrict__ out,
                      const half8* __restrict__ /*weights*/,
                      const half8* __restrict__ /*biases*/,
                      int numLines,
                      int lineSize,
                      int /*baseLine*/,
                      const PostOpsNDParams* p)
{
    const half x = (half) *reinterpret_cast<const float*>(p->params);

    const half8 xVal = (x <= (half)0.0) ? (half8)X_MAX : (half8)x;

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    int i = 0;

    for (; i < numVectors - 7; i += 8)
    {
        out[i + 0] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 0], xVal);
        out[i + 1] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 1], xVal);
        out[i + 2] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 2], xVal);
        out[i + 3] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 3], xVal);
        out[i + 4] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 4], xVal);
        out[i + 5] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 5], xVal);
        out[i + 6] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 6], xVal);
        out[i + 7] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i + 7], xVal);
    }

    for (; i < numVectors; ++i)
        out[i] = __builtin_shave_cmu_clamp0_f16_rr_half8(in[i], xVal);
}

static void scale_fp16_inner(const half8* __restrict__ in,
                             half8*       __restrict__ out,
                             const half8* __restrict__ weights,
                             const half8* __restrict__ /*biases*/,
                             int numLines,
                             int lineSize,
                             int /*baseLine*/,
                             const PostOpsNDParams* /*p*/)
{
    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int axis_i = 0;

    for (; axis_i < lineVectors - 3; axis_i += 4)
    {
        const half8* __restrict__ rin = in + axis_i;
        half8* __restrict__ rout = out + axis_i;

        half8 rw0 = weights[axis_i + 0];
        half8 rw1 = weights[axis_i + 1];
        half8 rw2 = weights[axis_i + 2];
        half8 rw3 = weights[axis_i + 3];

        int i = 0;

        if (numLines > 2)
        {
            half8 rr00 = rin[0] * rw0;
            half8 rr01 = rin[1] * rw1;
            half8 rr02 = rin[2] * rw2;
            half8 rr03 = rin[3] * rw3;
            rin += lineVectors;

            half8 rr10 = rin[0];
            half8 rr11 = rin[1];
            half8 rr12 = rin[2] * rw2;
            half8 rr13 = rin[3] * rw3;
            rin += lineVectors;

#pragma clang loop unroll_count(4)
            for (; i < numLines - 2; ++i)
            {
                half8 rr20 = rin[0];
                half8 rr21 = rin[1];
                half8 rr22 = rin[2];
                half8 rr23 = rin[3];
                rin += lineVectors;

                rout[0] = rr00;
                rout[1] = rr01;
                rout[2] = rr02;
                rout[3] = rr03;
                rout += lineVectors;

                rr00 = rr10 * rw0;
                rr01 = rr11 * rw1;
                rr02 = rr12;
                rr03 = rr13;

                rr10 = rr20;
                rr11 = rr21;
                rr12 = rr22 * rw2;
                rr13 = rr23 * rw3;
            }

            rin -= 2 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = rin[0] * rw0;
            rout[1] = rin[1] * rw1;
            rout[2] = rin[2] * rw2;
            rout[3] = rin[3] * rw3;
            rin += lineVectors;
            rout += lineVectors;
        }
    }

    for (; axis_i < lineVectors; ++axis_i)
    {
        const half8* __restrict__ rin = in + axis_i;
        half8* __restrict__ rout = out + axis_i;

        half8 rw0 = weights[axis_i];

        int i = 0;

        if (numLines > 2)
        {
            half8 rr0 = rin[0] * rw0;
            rin += lineVectors;

            half8 rr1 = rin[0];
            rin += lineVectors;

#pragma clang loop unroll_count(8)
            for (; i < numLines - 2; ++i)
            {
                half8 rr2 = rin[0];
                rin += lineVectors;

                rout[0] = rr0;
                rout += lineVectors;

                rr0 = rr1 * rw0;

                rr1 = rr2;
            }

            rin -= 2 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = rin[0] * rw0;
            rin  += lineVectors;
            rout += lineVectors;
        }
    }
}

static void scale_fp16_outer(const half8* __restrict__ in,
                             half8*       __restrict__ out,
                             const half8* __restrict__ _weights,
                             const half8* __restrict__ /*biases*/,
                             int numLines,
                             int lineSize,
                             int baseLine,
                             const PostOpsNDParams* p)
{
    const half* weights = (const half*)_weights;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < numLines - 7; line_i += 8)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;
        int axis_i1 = ((baseLine + line_i + 1) / p->axisGran) % p->axisDim;
        int axis_i2 = ((baseLine + line_i + 2) / p->axisGran) % p->axisDim;
        int axis_i3 = ((baseLine + line_i + 3) / p->axisGran) % p->axisDim;
        int axis_i4 = ((baseLine + line_i + 4) / p->axisGran) % p->axisDim;
        int axis_i5 = ((baseLine + line_i + 5) / p->axisGran) % p->axisDim;
        int axis_i6 = ((baseLine + line_i + 6) / p->axisGran) % p->axisDim;
        int axis_i7 = ((baseLine + line_i + 7) / p->axisGran) % p->axisDim;

        half8 rw0 = (half8) weights[axis_i0];
        half8 rw1 = (half8) weights[axis_i1];
        half8 rw2 = (half8) weights[axis_i2];
        half8 rw3 = (half8) weights[axis_i3];
        half8 rw4 = (half8) weights[axis_i4];
        half8 rw5 = (half8) weights[axis_i5];
        half8 rw6 = (half8) weights[axis_i6];
        half8 rw7 = (half8) weights[axis_i7];

        if (lineVectors >= 1)
        {
            int i = 0;

            half8 rr00 = in[(line_i + 0) * lineVectors + i] * rw0;
            half8 rr01 = in[(line_i + 1) * lineVectors + i] * rw1;
            half8 rr02 = in[(line_i + 2) * lineVectors + i] * rw2;
            half8 rr03 = in[(line_i + 3) * lineVectors + i] * rw3;
            half8 rr04 = in[(line_i + 4) * lineVectors + i];
            half8 rr05 = in[(line_i + 5) * lineVectors + i];
            half8 rr06 = in[(line_i + 6) * lineVectors + i];
            half8 rr07 = in[(line_i + 7) * lineVectors + i];

            for (i = 1; i < lineVectors; ++i)
            {
                half8 rr10 = in[(line_i + 0) * lineVectors + i] * rw0;
                half8 rr11 = in[(line_i + 1) * lineVectors + i] * rw1;
                half8 rr12 = in[(line_i + 2) * lineVectors + i] * rw2;
                half8 rr13 = in[(line_i + 3) * lineVectors + i] * rw3;
                half8 rr14 = in[(line_i + 4) * lineVectors + i];
                half8 rr15 = in[(line_i + 5) * lineVectors + i];
                half8 rr16 = in[(line_i + 6) * lineVectors + i];
                half8 rr17 = in[(line_i + 7) * lineVectors + i];

                out[(line_i + 0) * lineVectors + (i - 1)] = rr00;
                out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
                out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
                out[(line_i + 3) * lineVectors + (i - 1)] = rr03;
                out[(line_i + 4) * lineVectors + (i - 1)] = rr04 * rw4;
                out[(line_i + 5) * lineVectors + (i - 1)] = rr05 * rw5;
                out[(line_i + 6) * lineVectors + (i - 1)] = rr06 * rw6;
                out[(line_i + 7) * lineVectors + (i - 1)] = rr07 * rw7;

                rr00 = rr10;
                rr01 = rr11;
                rr02 = rr12;
                rr03 = rr13;
                rr04 = rr14;
                rr05 = rr15;
                rr06 = rr16;
                rr07 = rr17;
            }

            out[(line_i + 0) * lineVectors + (i - 1)] = rr00;
            out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
            out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
            out[(line_i + 3) * lineVectors + (i - 1)] = rr03;
            out[(line_i + 4) * lineVectors + (i - 1)] = rr04 * rw4;
            out[(line_i + 5) * lineVectors + (i - 1)] = rr05 * rw5;
            out[(line_i + 6) * lineVectors + (i - 1)] = rr06 * rw6;
            out[(line_i + 7) * lineVectors + (i - 1)] = rr07 * rw7;
        }
    }

    for (; line_i < numLines; ++line_i)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;

        half8 rw0 = (half8) weights[axis_i0];

        for (int i = 0; i < lineVectors; ++i)
        {
            half8 rr0 = in[(line_i + 0) * lineVectors + i];

            out[(line_i + 0) * lineVectors + i] = rr0 * rw0;
        }
    }
}

static void scale_shift_fp16_inner(const half8* __restrict__ in,
                                   half8*       __restrict__ out,
                                   const half8* __restrict__ weights,
                                   const half8* __restrict__ biases,
                                   int numLines,
                                   int lineSize,
                                   int /*baseLine*/,
                                   const PostOpsNDParams* /*p*/)
{
    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int axis_i = 0;

    for (; axis_i < lineVectors - 3; axis_i += 4)
    {
        const half8* __restrict__ rin = in + axis_i;
        half8* __restrict__ rout = out + axis_i;

        half8 rw0 = weights[axis_i + 0];
        half8 rw1 = weights[axis_i + 1];
        half8 rw2 = weights[axis_i + 2];
        half8 rw3 = weights[axis_i + 3];

        half8 rb0 = biases[axis_i + 0];
        half8 rb1 = biases[axis_i + 1];
        half8 rb2 = biases[axis_i + 2];
        half8 rb3 = biases[axis_i + 3];

        int i = 0;

        if (numLines > 2)
        {
            half8 rr00 = rin[0] * rw0 + rb0;
            half8 rr01 = rin[1] * rw1 + rb1;
            half8 rr02 = rin[2] * rw2 + rb2;
            half8 rr03 = rin[3] * rw3 + rb3;
            rin += lineVectors;

            half8 rr10 = rin[0];
            half8 rr11 = rin[1];
            half8 rr12 = rin[2] * rw2 + rb2;
            half8 rr13 = rin[3] * rw3 + rb3;
            rin += lineVectors;

#pragma clang loop unroll_count(3)
            for (; i < numLines - 2; ++i)
            {
                half8 rr20 = rin[0];
                half8 rr21 = rin[1];
                half8 rr22 = rin[2];
                half8 rr23 = rin[3];
                rin += lineVectors;

                rout[0] = rr00;
                rout[1] = rr01;
                rout[2] = rr02;
                rout[3] = rr03;
                rout += lineVectors;

                rr00 = rr10 * rw0 + rb0;
                rr01 = rr11 * rw1 + rb1;
                rr02 = rr12;
                rr03 = rr13;

                rr10 = rr20;
                rr11 = rr21;
                rr12 = rr22 * rw2 + rb2;
                rr13 = rr23 * rw3 + rb3;
            }

            rin -= 2 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = rin[0] * rw0 + rb0;
            rout[1] = rin[1] * rw1 + rb1;
            rout[2] = rin[2] * rw2 + rb2;
            rout[3] = rin[3] * rw3 + rb3;
            rin += lineVectors;
            rout += lineVectors;
        }
    }

    for (; axis_i < lineVectors; ++axis_i)
    {
        const half8* __restrict__ rin = in + axis_i;
        half8* __restrict__ rout = out + axis_i;

        half8 rw0 = weights[axis_i];
        half8 rb0 = biases[axis_i];

        int i = 0;

        if (numLines > 3)
        {
            half8 rr0 = rin[0] * rw0 + rb0;
            rin += lineVectors;

            half8 rr1 = rin[0] * rw0;
            rin += lineVectors;

            half8 rr2 = rin[0];
            rin += lineVectors;

#pragma clang loop unroll_count(4)
            for (; i < numLines - 3; ++i)
            {
                half8 rr3 = rin[0];
                rin += lineVectors;

                rout[0] = rr0;
                rout += lineVectors;

                rr0 = rr1 + rb0;
                rr1 = rr2 * rw0;

                rr2 = rr3;
            }

            rin -= 3 * lineVectors;
        }

        for (; i < numLines; ++i)
        {
            rout[0] = rin[0] * rw0 + rb0;
            rin  += lineVectors;
            rout += lineVectors;
        }
    }
}

static void scale_shift_fp16_outer(const half8* __restrict__ in,
                                   half8*       __restrict__ out,
                                   const half8* __restrict__ _weights,
                                   const half8* __restrict__ _biases,
                                   int numLines,
                                   int lineSize,
                                   int baseLine,
                                   const PostOpsNDParams* p)
{
    const half* weights = (const half*)_weights;
    const half* biases = (const half*)_biases;

    const int lineVectors = DIVR(lineSize, VECTOR_SIZE);

    int line_i = 0;

    for (; line_i < numLines - 7; line_i += 8)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;
        int axis_i1 = ((baseLine + line_i + 1) / p->axisGran) % p->axisDim;
        int axis_i2 = ((baseLine + line_i + 2) / p->axisGran) % p->axisDim;
        int axis_i3 = ((baseLine + line_i + 3) / p->axisGran) % p->axisDim;
        int axis_i4 = ((baseLine + line_i + 4) / p->axisGran) % p->axisDim;
        int axis_i5 = ((baseLine + line_i + 5) / p->axisGran) % p->axisDim;
        int axis_i6 = ((baseLine + line_i + 6) / p->axisGran) % p->axisDim;
        int axis_i7 = ((baseLine + line_i + 7) / p->axisGran) % p->axisDim;

        half8 rw0 = (half8) weights[axis_i0];
        half8 rw1 = (half8) weights[axis_i1];
        half8 rw2 = (half8) weights[axis_i2];
        half8 rw3 = (half8) weights[axis_i3];
        half8 rw4 = (half8) weights[axis_i4];
        half8 rw5 = (half8) weights[axis_i5];
        half8 rw6 = (half8) weights[axis_i6];
        half8 rw7 = (half8) weights[axis_i7];

        half8 rb0 = (half8) biases[axis_i0];
        half8 rb1 = (half8) biases[axis_i1];
        half8 rb2 = (half8) biases[axis_i2];
        half8 rb3 = (half8) biases[axis_i3];
        half8 rb4 = (half8) biases[axis_i4];
        half8 rb5 = (half8) biases[axis_i5];
        half8 rb6 = (half8) biases[axis_i6];
        half8 rb7 = (half8) biases[axis_i7];

        if (lineVectors >= 1)
        {
            int i = 0;

            half8 rr00 = in[(line_i + 0) * lineVectors + i];
            half8 rr01 = in[(line_i + 1) * lineVectors + i] * rw1 + rb1;
            half8 rr02 = in[(line_i + 2) * lineVectors + i] * rw2 + rb2;
            half8 rr03 = in[(line_i + 3) * lineVectors + i] * rw3 + rb3;
            half8 rr04 = in[(line_i + 4) * lineVectors + i] * rw4 + rb4;
            half8 rr05 = in[(line_i + 5) * lineVectors + i] * rw5 + rb5;
            half8 rr06 = in[(line_i + 6) * lineVectors + i] * rw6 + rb6;
            half8 rr07 = in[(line_i + 7) * lineVectors + i] * rw7 + rb7;

            for (i = 1; i < lineVectors; ++i)
            {
                half8 rr10 = in[(line_i + 0) * lineVectors + i];
                half8 rr11 = in[(line_i + 1) * lineVectors + i] * rw1 + rb1;
                half8 rr12 = in[(line_i + 2) * lineVectors + i] * rw2 + rb2;
                half8 rr13 = in[(line_i + 3) * lineVectors + i] * rw3 + rb3;
                half8 rr14 = in[(line_i + 4) * lineVectors + i] * rw4 + rb4;
                half8 rr15 = in[(line_i + 5) * lineVectors + i] * rw5 + rb5;
                half8 rr16 = in[(line_i + 6) * lineVectors + i] * rw6 + rb6;
                half8 rr17 = in[(line_i + 7) * lineVectors + i] * rw7 + rb7;

                out[(line_i + 0) * lineVectors + (i - 1)] = rr00 * rw0 + rb0;
                out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
                out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
                out[(line_i + 3) * lineVectors + (i - 1)] = rr03;
                out[(line_i + 4) * lineVectors + (i - 1)] = rr04;
                out[(line_i + 5) * lineVectors + (i - 1)] = rr05;
                out[(line_i + 6) * lineVectors + (i - 1)] = rr06;
                out[(line_i + 7) * lineVectors + (i - 1)] = rr07;

                rr00 = rr10;
                rr01 = rr11;
                rr02 = rr12;
                rr03 = rr13;
                rr04 = rr14;
                rr05 = rr15;
                rr06 = rr16;
                rr07 = rr17;
            }

            out[(line_i + 0) * lineVectors + (i - 1)] = rr00 * rw0 + rb0;
            out[(line_i + 1) * lineVectors + (i - 1)] = rr01;
            out[(line_i + 2) * lineVectors + (i - 1)] = rr02;
            out[(line_i + 3) * lineVectors + (i - 1)] = rr03;
            out[(line_i + 4) * lineVectors + (i - 1)] = rr04;
            out[(line_i + 5) * lineVectors + (i - 1)] = rr05;
            out[(line_i + 6) * lineVectors + (i - 1)] = rr06;
            out[(line_i + 7) * lineVectors + (i - 1)] = rr07;
        }
    }

    for (; line_i < numLines; ++line_i)
    {
        int axis_i0 = ((baseLine + line_i + 0) / p->axisGran) % p->axisDim;

        half8 rw0 = (half8) weights[axis_i0];

        half8 rb0 = (half8) biases[axis_i0];

        for (int i = 0; i < lineVectors; ++i)
        {
            half8 rr0 = in[(line_i + 0) * lineVectors + i];

            out[(line_i + 0) * lineVectors + i] = rr0 * rw0 + rb0;
        }
    }
}

static void sigmoid_fp16(const half8* __restrict__ in,
                         half8*       __restrict__ out,
                         const half8* __restrict__ /*weights*/,
                         const half8* __restrict__ /*biases*/,
                         int numLines,
                         int lineSize,
                         int /*baseLine*/,
                         const PostOpsNDParams* /*p*/)
{
    // Compute sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x/ln(2)))
    const uint16_t negative_inv_ln2 = 0xbdc6;
    const half negative_inv_ln2_h = *(const half*)&negative_inv_ln2;
    const half one = (half)1.0f;

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

#pragma clang loop unroll_count(8)
    for (int i = 0; i < numVectors; ++i)
    {
        out[i] = in[i] * negative_inv_ln2_h;
        exp2_vec(out[i], out[i]);
    }

#pragma clang loop unroll_count(8)
    for (int i = 0; i < numVectors; ++i)
    {
        out[i] = one / (one + (out[i]));
    }
}

static void tanh_fp16(const half8* __restrict__ in,
                      half8*       __restrict__ out,
                      const half8* __restrict__ /*weights*/,
                      const half8* __restrict__ /*biases*/,
                      int numLines,
                      int lineSize,
                      int /*baseLine*/,
                      const PostOpsNDParams* /*p*/)
{
    // Clamp the input to avoid fp16 precision overflow when computing exp.
    // This should not affect the results
    const half8 upper_bound =   5.5f;
    const half8 lower_bound = -10.5f;

    // Compute tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // = (2^(2x/ln(2)) - 1) / (2^(2x/ln(2)) + 1)
    const uint16_t inv_ln2_mul_2 = 0x41c5;
    const half inv_ln2_mul_2_h = *(const half*)&inv_ln2_mul_2;
    const half one = (half)1.0f;

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

#pragma clang loop unroll_count(8) // 64/99/6 = 1.546875
    for (int i = 0; i < numVectors; ++i)
    {
        out[i] = __builtin_shave_cmu_clampab_f16_rrr_half8(in[i], lower_bound, upper_bound);
        out[i] = out[i] * inv_ln2_mul_2_h;
        exp2_vec(out[i], out[i]);
    }

#pragma clang loop unroll_count(5)
    for (int i = 0; i < numVectors; ++i)
    {
        out[i] = (out[i] - one) / (out[i] + one);
    }
}

void log_fp16(const half8* __restrict__ in,
              half8*       __restrict__ out,
              const half8* __restrict__ /*weights*/,
              const half8* __restrict__ /*biases*/,
              int numLines,
              int lineSize,
              int /*baseLine*/,
              const PostOpsNDParams* /*p*/)
{
    const unsigned short ln2 = 0x398c;
    const half ln2_h = *reinterpret_cast<const half *>(&ln2);

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

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

void exp_fp16(const half8* __restrict__ in,
              half8*       __restrict__ out,
              const half8* __restrict__ /*weights*/,
              const half8* __restrict__ /*biases*/,
              int numLines,
              int lineSize,
              int /*baseLine*/,
              const PostOpsNDParams* /*p*/)
{
    const uint16_t inv_ln2 = 0x3dc5;
    const half inv_ln2_h = *(const half*)&inv_ln2;

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    #pragma clang loop unroll_count(8)
    for(s32 i = 0; i < numVectors; ++i)
    {
        half8 exp_x = in[i] * inv_ln2_h;
        exp2_vec(exp_x, exp_x);
        out[i] = exp_x;
    }
}

void floor_fp16(const half8* __restrict__ in,
                half8*       __restrict__ out,
                const half8* __restrict__ /*weights*/,
                const half8* __restrict__ /*biases*/,
                int numLines,
                int lineSize,
                int /*baseLine*/,
                const PostOpsNDParams* /*p*/)
{
    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    #pragma clang loop unroll_count(4)
    for(s32 i = 0; i < numVectors; ++i)
    {
        out[i] = floorh8(in[i]);
    }
}

void ceil_fp16(const half8* __restrict__ in,
               half8*       __restrict__ out,
               const half8* __restrict__ /*weights*/,
               const half8* __restrict__ /*biases*/,
               int numLines,
               int lineSize,
               int /*baseLine*/,
               const PostOpsNDParams* /*p*/)
{
    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    #pragma clang loop unroll_count(4)
    for(s32 i = 0; i < numVectors; ++i)
    {
        out[i] = ceilh8(in[i]);
    }
}

void round_fp16(const half8* __restrict__ in,
                half8*       __restrict__ out,
                const half8* __restrict__ /*weights*/,
                const half8* __restrict__ /*biases*/,
                int numLines,
                int lineSize,
                int /*baseLine*/,
                const PostOpsNDParams* p)
{
    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    const t_RoundLayerParams* params = reinterpret_cast<const t_RoundLayerParams*>(p->params);

    #pragma clang loop unroll_count(4)
    for(s32 i = 0; i < numVectors; ++i)
    {
        out[i] = roundh8(in[i], params->mode);
    }
}

static void erf_fp16(const half8* __restrict__ in,
                     half8*       __restrict__ out,
                     const half8* __restrict__ /*weights*/,
                     const half8* __restrict__ /*biases*/,
                     int numLines,
                     int lineSize,
                     int /*baseLine*/,
                     const PostOpsNDParams* /*p*/)
{
    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    #pragma clang loop unroll_count(4)
    for(s32 i = 0; i < numVectors; ++i)
    {
        out[i] = erfh8(in[i]);
    }
}

static void mish_fp16(const half8* __restrict__ in,
                      half8*       __restrict__ out,
                      const half8* __restrict__ /*weights*/,
                      const half8* __restrict__ /*biases*/,
                      int numLines,
                      int lineSize,
                      int /*baseLine*/,
                      const PostOpsNDParams* /*p*/)
{
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

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);
    if (numVectors == 0)
        return;

    half8 tmp = in[0];

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
        out[i - 1] = in[i - 1] * tmp;

        tmp = in[i];

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

    out[numVectors - 1] = in[numVectors - 1] * tmp;
}

static void gelu_fp16(const half8* __restrict__ in,
                      half8*       __restrict__ out,
                      const half8* __restrict__ /*weights*/,
                      const half8* __restrict__ /*biases*/,
                      int numLines,
                      int lineSize,
                      int /*baseLine*/,
                      const PostOpsNDParams* /*p*/)
{
    // Gelu(x) = x * P(X <=x) = x * F(x) = 0.5 * x * (1 + erf(x / sqrt(2))
    // which in turn is approximated to
    // 0.5 * x * ( 1 + tanh[sqrt(2 / pi) * (x + 0.044715 * x^3)])
    // == 0.5 * x * (2 * exp(c) / (exp(c) + exp(-c)), where c = sqrt(2 / pi) * (x + 0.044715 * x^3)
    // == x / (1 + exp(-2 * sqrt(2 / pi) * x * (1 + 0.044715 * x^2)))

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

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

static void softplus_fp16(const half8* __restrict__ in,
                          half8*       __restrict__ out,
                          const half8* __restrict__ /*weights*/,
                          const half8* __restrict__ /*biases*/,
                          int numLines,
                          int lineSize,
                          int /*baseLine*/,
                          const PostOpsNDParams* /*p*/)
{
// SoftPlus(x) = log(1 + exp(x))
//             = log1p(exp(x)) | x << 0
//             = x + log1p(exp(-x)) | x >> 0

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    const float8 one = (float8)1.0f;

    int i = 0;
    if (numVectors >= 2)
    {
        float8 x0 = mvuConvert_float8(in[i + 0]);
        float8 x1 = mvuConvert_float8(in[i + 1]);
        float8 tmp0 = expf8(x0);
        float8 tmp1 = expf8(x1);
        for (i = 2; i < numVectors - 1; i += 2)
        {
            float8 x2 = mvuConvert_float8(in[i + 0]);
            float8 x3 = mvuConvert_float8(in[i + 1]);
            float8 tmp2 = expf8(x2);
            float8 tmp3 = expf8(x3);
            float8 res0 = logf8(one + tmp0);
            float8 res1 = logf8(one + tmp1);
            out[i - 2] = mvuConvert_half8(res0);
            out[i - 1] = mvuConvert_half8(res1);
            tmp0 = tmp2;
            tmp1 = tmp3;
        }
        float8 res0 = logf8(one + tmp0);
        float8 res1 = logf8(one + tmp1);
        out[i - 2] = mvuConvert_half8(res0);
        out[i - 1] = mvuConvert_half8(res1);
    }
    for (; i < numVectors; ++i)
    {
        float8 x = mvuConvert_float8(in[i]);
        float8 res = logf8(one + expf8(x));
        out[i] = mvuConvert_half8(res);
    }
}

static void swish_fp16(const half8* __restrict__ in,
                       half8*       __restrict__ out,
                       const half8* __restrict__ /*weights*/,
                       const half8* __restrict__ /*biases*/,
                       int numLines,
                       int lineSize,
                       int /*baseLine*/,
                       const PostOpsNDParams* p)
{
    // Swish(x) = x / (1 + exp(-beta * x))

    const t_SwishLayerParams* swishParams = reinterpret_cast<const t_SwishLayerParams*>(p->params);
    const half beta = (half)swishParams->beta;

    uint16_t inv_ln2 = 0x3dc5;
    half inv_ln2_h = *(const half*)&inv_ln2;

    const half8 one = (half8)1.0h;
    const half8 nbeta_inv_ln2 = (half8)(-beta * inv_ln2_h);

    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

#pragma clang loop unroll_count(16)
    for(int i = 0; i < numVectors; ++i)
    {
        const half8 x = in[i];

        half8 exponent = __builtin_shave_vau_mul_f16_rr(nbeta_inv_ln2, x);
        exp2_vec(exponent, exponent);

        half8 denom = __builtin_shave_vau_add_f16_rr(exponent, one);
        half8 inv_denom = 1 / denom;

        half8 res = __builtin_shave_vau_mul_f16_rr(x, inv_denom);

        out[i] = res;
    }
}

static void hswish_fp16(const half8* __restrict__ in,
                        half8*       __restrict__ out,
                        const half8* __restrict__ /*weights*/,
                        const half8* __restrict__ /*biases*/,
                        int numLines,
                        int lineSize,
                        int /*baseLine*/,
                        const PostOpsNDParams* /*params*/)
{
    // numVectors shows how many half8 vectors we can split the original data
    // but no more than the CMX memory size allows
    const int numVectors = DIVR(numLines * lineSize, VECTOR_SIZE);

    // HSwish(x) = x * ReLU6(x + 3) / 6;
    const half8 addVal3 = 3.0f;
    const half8 maxVal6 = 6.0f;

    int i = 0;

    if (numVectors >= 8)
    {
        half8 r00 = in[0];
        half8 r01 = in[1];
        half8 r02 = in[2];
        half8 r03 = in[3];
        half8 r04 = in[4];
        half8 r05 = in[5];
        half8 r06 = in[6];
        half8 r07 = in[7];
        in += 8;

        for (i = 8; i < numVectors - 7; i += 8)
        {
            half8 r10 = in[0];
            half8 r11 = in[1];
            half8 r12 = in[2];
            half8 r13 = in[3];
            half8 r14 = in[4];
            half8 r15 = in[5];
            half8 r16 = in[6];
            half8 r17 = in[7];
            in += 8;

            out[0] = __builtin_shave_cmu_clamp0_f16_rr_half8(r00 + addVal3, maxVal6) * r00 / maxVal6;
            out[1] = __builtin_shave_cmu_clamp0_f16_rr_half8(r01 + addVal3, maxVal6) * r01 / maxVal6;
            out[2] = __builtin_shave_cmu_clamp0_f16_rr_half8(r02 + addVal3, maxVal6) * r02 / maxVal6;
            out[3] = __builtin_shave_cmu_clamp0_f16_rr_half8(r03 + addVal3, maxVal6) * r03 / maxVal6;
            out[4] = __builtin_shave_cmu_clamp0_f16_rr_half8(r04 + addVal3, maxVal6) * r04 / maxVal6;
            out[5] = __builtin_shave_cmu_clamp0_f16_rr_half8(r05 + addVal3, maxVal6) * r05 / maxVal6;
            out[6] = __builtin_shave_cmu_clamp0_f16_rr_half8(r06 + addVal3, maxVal6) * r06 / maxVal6;
            out[7] = __builtin_shave_cmu_clamp0_f16_rr_half8(r07 + addVal3, maxVal6) * r07 / maxVal6;
            out += 8;

            r00 = r10;
            r01 = r11;
            r02 = r12;
            r03 = r13;
            r04 = r14;
            r05 = r15;
            r06 = r16;
            r07 = r17;
        }

        out[0] = __builtin_shave_cmu_clamp0_f16_rr_half8(r00 + addVal3, maxVal6) * r00 / maxVal6;
        out[1] = __builtin_shave_cmu_clamp0_f16_rr_half8(r01 + addVal3, maxVal6) * r01 / maxVal6;
        out[2] = __builtin_shave_cmu_clamp0_f16_rr_half8(r02 + addVal3, maxVal6) * r02 / maxVal6;
        out[3] = __builtin_shave_cmu_clamp0_f16_rr_half8(r03 + addVal3, maxVal6) * r03 / maxVal6;
        out[4] = __builtin_shave_cmu_clamp0_f16_rr_half8(r04 + addVal3, maxVal6) * r04 / maxVal6;
        out[5] = __builtin_shave_cmu_clamp0_f16_rr_half8(r05 + addVal3, maxVal6) * r05 / maxVal6;
        out[6] = __builtin_shave_cmu_clamp0_f16_rr_half8(r06 + addVal3, maxVal6) * r06 / maxVal6;
        out[7] = __builtin_shave_cmu_clamp0_f16_rr_half8(r07 + addVal3, maxVal6) * r07 / maxVal6;
        out += 8;
    }

    for (; i < numVectors; ++i)
    {
        *out++ = __builtin_shave_cmu_clamp0_f16_rr_half8(*in + addVal3, maxVal6) * *in / maxVal6;
        in++;
    }
}

static void dma_start_3d(DmaAlShave& dma, const void* src, void* dst, u32 byteLength,
                          u32 srcWidth, u32 dstWidth, u32 srcStride, u32 dstStride,
                          u32 numPlanes, u32 srcPlaneStride, u32 dstPlaneStride)
{
    if (((byteLength % srcWidth) == 0) && (srcStride * byteLength == srcPlaneStride * srcWidth) &&
        ((byteLength % dstWidth) == 0) && (dstStride * byteLength == dstPlaneStride * dstWidth))
    {
        byteLength *= numPlanes;
        numPlanes = 1;
    }
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;
    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
}

static void dma_start_2d(DmaAlShave& dma, const void* src, void* dst, u32 byteLength,
                          u32 srcWidth, u32 dstWidth, u32 srcStride, u32 dstStride)
{
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;
    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
}

static void postOps_ND_core_3D(Operation op, const PostOpsNDParams* p)
{
    DmaAlShave dma1;
#if defined(ASYNC_PIPELINE)
    DmaAlShave dma2;
#endif

    const int hasAxis = (p->axis >= 0) ? 1 : 0;
    const int hasAxisData = (p->weights || p->biases) ? hasAxis : 0;

    const int planesLimit = p->start + p->toProcess;
    int currentPlane = p->start;

    const int maxPlanes = MIN(p->dims[0], DmaAlShave::max_3D_planes);
    const int lineSize = hasAxisData ? ALIGN_TO_MULTIPLE(VECTOR_SIZE, p->dim0) : p->dim0;
    const int planeSize = lineSize * p->dim1;

    fp16* cmx = (fp16*) p->cmxSlice;
    const int cmxSize = p->cmxSize;

// TODO: code enclosed in #if..endif isn't compiled for KMB yet
#if defined(ASYNC_PIPELINE)
    const int cmxPlanes = MIN(maxPlanes, (cmxSize - VECTOR_SIZE * 3) / (planeSize * INPUT_BPP * 3));
    const int minSteps = DIVR(planesLimit - currentPlane, cmxPlanes);
    if (minSteps > 3) // async mode
    {
        nnLog(MVLOG_DEBUG, "%d # postOps_ND_core_3D : async\n", scGetShaveNumber());

        const int bufferSize = ALIGN_TO_MULTIPLE(VECTOR_SIZE, planeSize * cmxPlanes);

        fp16* planeBuffer_0 = cmx; cmx += bufferSize;
        fp16* planeBuffer_1 = cmx; cmx += bufferSize;
        fp16* planeBuffer_2 = cmx;

        const u8 *in_0 = 0, *in_1 = 0, *in_2 = 0;
        u8 *out_0 = 0, *out_1 = 0, *out_2 = 0;

        int numPlanes_0 = 0, numPlanes_1 = 0, numPlanes_2 = 0;
        int currentPlane_1 = 0;

        // action: finite state machine
        //   -2 -1 0 1 2 3:exit
        // states:
        //   -2 -1 0       : read(0)
        //      -1 0 1     : process(1)
        //         0 1 2   : write(2)
        //               3 : exit
        // transitions:
        //   0 => check (currentLine < linesLimit)
        //   else ++ (i.e. next)

        DmaAlShave* last = nullptr;

        int32_t setCoords[MAX_ND_DIMS];
        subspace::getCoord(currentPlane, p->dims, p->ndims, setCoords);

        int action = -2;
        do {
            if (action <= 0)
            {
                numPlanes_0 = MIN(cmxPlanes, p->dims[0] - setCoords[0]);

                unsigned inOffset, outOffset;
                subspace::getOffsetsU8(setCoords, p->istrides, p->ostrides, p->ndims, inOffset, outOffset);

                in_0 = (u8*)(p->input) + inOffset;
                out_0 = (u8*)(p->output) + outOffset;
            }

            if (last)
            {
                last->wait();
                last = nullptr;
            }
            {
                DmaAlShave* prev = nullptr;
                if (action >= 0)
                {
                    prev = last = &dma2;
#error Review the usage of dma_start_3d in place of dma_create_3d
                    dma_start_3d(*last, planeBuffer_2, out_2, p->dim0 * p->dim1 * INPUT_BPP,
                                  p->dim0 * INPUT_BPP, p->dim0 * INPUT_BPP, lineSize * INPUT_BPP, p->outStride1,
                                  numPlanes_2, planeSize * INPUT_BPP, p->ostrides[0]);
                }
                if (action <= 0)
                {
                    last = &dma1;
#error Review the usage of dma_start_3d in place of dma_create_3d
                    dma_start_3d(*last, in_0, planeBuffer_0, p->dim0 * p->dim1 * INPUT_BPP,
                                  p->dim0 * INPUT_BPP, p->dim0 * INPUT_BPP, p->inStride1, lineSize * INPUT_BPP,
                                  numPlanes_0, p->istrides[0], planeSize * INPUT_BPP);
                    if (prev)
                    {
                        last->append(*prev);
                        prev = nullptr;
                    }
                }
            }

            if ((action >= -1) && (action <= 1))
            {
                const int numLines_1 = numPlanes_1 * p->dim1;
                const int currentLine_1 = currentPlane_1 * p->dim1;
                op((half8*)planeBuffer_1, (half8*)planeBuffer_1, (half8*)p->weights, (half8*)p->biases, numLines_1, lineSize, currentLine_1, p);
            }

            SWAP3(planeBuffer_0, planeBuffer_1, planeBuffer_2);
            out_2 = out_1; out_1 = out_0; //SWAP3(out_0, out_1, out_2);
            in_2 = in_1; in_1 = in_0; //SWAP3(in_0, in_1, in_2);
            numPlanes_2 = numPlanes_1; numPlanes_1 = numPlanes_0;
            currentPlane_1 = currentPlane;

            if (action <= 0)
            {
                subspace::incrementNCoord(setCoords, p->dims, p->ndims, numPlanes_0);
                currentPlane += numPlanes_0;
            }

            action += (action == 0) ? !(currentPlane < planesLimit) : 1;
        } while (action <= 2);

        if (last)
        {
            last->wait();
            last = nullptr;
        }
    }
    else // sync mode
#endif // ASYNC_PIPELINE
    {
        nnLog(MVLOG_DEBUG, "%d # postOps_ND_core_3D : >> sync\n", scGetShaveNumber());

        const int cmxPlanes = MIN(maxPlanes, cmxSize / (planeSize * INPUT_BPP));

        fp16* planeBuffer = cmx;

        int32_t setCoords[MAX_ND_DIMS];
        subspace::getCoord(currentPlane, p->dims, p->ndims, setCoords);

        while (currentPlane < planesLimit)
        {
            int numPlanes = MIN(cmxPlanes, p->dims[0] - setCoords[0]);
            //avoid processing unnessary data to save time
            numPlanes = MIN(numPlanes, planesLimit - currentPlane);
            unsigned inOffset, outOffset;
            subspace::getOffsetsU8(setCoords, p->istrides, p->ostrides, p->ndims, inOffset, outOffset);

            u8* in = (u8*)(p->input) + inOffset;
            u8* out = (u8*)(p->output) + outOffset;

            dma_start_3d(dma1, in, planeBuffer, p->dim0 * p->dim1 * INPUT_BPP,
                          p->dim0 * INPUT_BPP, p->dim0 * INPUT_BPP, p->inStride1, lineSize * INPUT_BPP,
                          numPlanes, p->istrides[0], planeSize * INPUT_BPP);
            dma1.wait();

            const int numLines = numPlanes * p->dim1;
            const int currentLine = currentPlane * p->dim1;
            op((half8*)planeBuffer, (half8*)planeBuffer, (half8*)p->weights, (half8*)p->biases, numLines, lineSize, currentLine, p);

            dma_start_3d(dma1, planeBuffer, out, p->dim0 * p->dim1 * INPUT_BPP,
                          p->dim0 * INPUT_BPP, p->dim0 * INPUT_BPP, lineSize * INPUT_BPP, p->outStride1,
                          numPlanes, planeSize * INPUT_BPP, p->ostrides[0]);
            dma1.wait();

            subspace::incrementNCoord(setCoords, p->dims, p->ndims, numPlanes);
            currentPlane += numPlanes;
        }

        nnLog(MVLOG_DEBUG, "%d # postOps_ND_core_3D : << sync\n", scGetShaveNumber());
    }
}

static void postOps_ND_core_2D(Operation op, const PostOpsNDParams* p)
{
    DmaAlShave dma1;
#if defined(ASYNC_PIPELINE)
    DmaAlShave dma2;
#endif

    const int hasAxis = (p->axis >= 0) ? 1 : 0;
    const int hasAxisData = (p->weights || p->biases) ? hasAxis : 0;

    const int linesLimit = p->start + p->toProcess;
    int currentLine = p->start;

    const int maxLines = p->dims[0];
    const int lineSize = hasAxisData ? ALIGN_TO_MULTIPLE(VECTOR_SIZE, p->dim0) : p->dim0;

    fp16* cmx = (fp16*) p->cmxSlice;
    const int cmxSize = p->cmxSize;

// TODO: code enclosed in #if..endif isn't compiled for KMB yet
#if defined(ASYNC_PIPELINE)
    const int cmxLines = MIN(maxLines, (cmxSize - VECTOR_SIZE * 3) / (lineSize * INPUT_BPP * 3));
    const int minSteps = DIVR(linesLimit - currentLine, cmxLines);
    if (minSteps > 3) // async mode
    {
        nnLog(MVLOG_DEBUG, "%d # postOps_ND_core_2D : async\n", scGetShaveNumber());

        const int bufferSize = ALIGN_TO_MULTIPLE(VECTOR_SIZE, lineSize * cmxLines);

        fp16* planeBuffer_0 = cmx; cmx += bufferSize;
        fp16* planeBuffer_1 = cmx; cmx += bufferSize;
        fp16* planeBuffer_2 = cmx;

        const u8 *in_0 = 0, *in_1 = 0, *in_2 = 0;
        u8 *out_0 = 0, *out_1 = 0, *out_2 = 0;

        int numLines_0 = 0, numLines_1 = 0, numLines_2 = 0;
        int currentLine_1 = 0;

        // action: finite state machine
        //   -2 -1 0 1 2 3:exit
        // states:
        //   -2 -1 0       : read(0)
        //      -1 0 1     : process(1)
        //         0 1 2   : write(2)
        //               3 : exit
        // transitions:
        //   0 => check (currentLine < linesLimit)
        //   else ++ (i.e. next)

        DmaAlShave* last = nullptr;

        int32_t setCoords[MAX_ND_DIMS];
        subspace::getCoord(currentLine, p->dims, p->ndims, setCoords);

        int action = -2;
        do {
            if (action <= 0)
            {
                numLines_0 = MIN(cmxLines, p->dims[0] - setCoords[0]);

                unsigned inOffset, outOffset;
                subspace::getOffsetsU8(setCoords, p->istrides, p->ostrides, p->ndims, inOffset, outOffset);

                in_0 = (u8*)(p->input) + inOffset;
                out_0 = (u8*)(p->output) + outOffset;
            }

            if (last)
            {
                last->wait();
                last = nullptr;
            }
            {
                DmaAlShave* prev = nullptr;
                if (action >= 0)
                {
                    prev = last = &dma2;
#error Review the usage of dma_start_2d in place of dma_create_2d
                    dma_start_2d(*last, planeBuffer_2, out_2, numLines_2 * p->dim0 * INPUT_BPP,
                                  p->dim0 * INPUT_BPP, p->dim0 * INPUT_BPP,
                                  lineSize * INPUT_BPP, p->ostrides[0]);
                }
                if (action <= 0)
                {
                    last = &dma1;
#error Review the usage of dma_start_2d in place of dma_create_2d
                    dma_start_2d(*last, in_0, planeBuffer_0, numLines_0 * p->dim0 * INPUT_BPP,
                                  p->dim0 * INPUT_BPP, p->dim0 * INPUT_BPP,
                                  p->istrides[0], lineSize * INPUT_BPP);
                    if (prev)
                    {
                        last->append(*prev);
                        prev = nullptr;
                    }
                }
            }

            if ((action >= -1) && (action <= 1))
            {
                op((half8*)planeBuffer_1, (half8*)planeBuffer_1, (half8*)p->weights, (half8*)p->biases, numLines_1, lineSize, currentLine_1, p);
            }

            SWAP3(planeBuffer_0, planeBuffer_1, planeBuffer_2);
            out_2 = out_1; out_1 = out_0; //SWAP3(out_0, out_1, out_2);
            in_2 = in_1; in_1 = in_0; //SWAP3(in_0, in_1, in_2);
            numLines_2 = numLines_1; numLines_1 = numLines_0;
            currentLine_1 = currentLine;

            if (action <= 0)
            {
                subspace::incrementNCoord(setCoords, p->dims, p->ndims, numLines_0);
                currentLine += numLines_0;
            }

            action += (action == 0) ? !(currentLine < linesLimit) : 1;
        } while (action <= 2);

        if (last)
        {
            last->wait();
            last = nullptr;
        }
    }
    else // sync mode
#endif // ASYNC_PIPELINE
    {
        nnLog(MVLOG_DEBUG, "%d # postOps_ND_core_2D : >> sync\n", scGetShaveNumber());

        const int cmxLines0 = MIN(maxLines, cmxSize / (lineSize * INPUT_BPP));
        const int maxLineSize0 = cmxSize / INPUT_BPP;
        const int maxLineSize1 = hasAxisData ? ((maxLineSize0 / VECTOR_SIZE) * VECTOR_SIZE) : maxLineSize0;

        const int cmxLines = (cmxLines0 < 1) ? 1 : cmxLines0;
        const int maxLineSize = (cmxLines0 < 1) ? maxLineSize1 : lineSize;

        fp16* planeBuffer = cmx;

        int32_t setCoords[MAX_ND_DIMS];
        subspace::getCoord(currentLine, p->dims, p->ndims, setCoords);

        while (currentLine < linesLimit)
        {
            int numLines = MIN(cmxLines, p->dims[0] - setCoords[0]);
            //avoid processing unnessary data to save time
            numLines = MIN(numLines, linesLimit - currentLine);

            unsigned inOffset, outOffset;
            subspace::getOffsetsU8(setCoords, p->istrides, p->ostrides, p->ndims, inOffset, outOffset);

            // note that "lineElem += stepElems" is more correct, but stepElems != maxLineSize only for the last iteration
            for (int lineElem = 0; lineElem < p->dim0; lineElem += maxLineSize)
            {
                const int stepElems = MIN(maxLineSize, p->dim0 - lineElem);
                const int numElems = hasAxisData ? ALIGN_TO_MULTIPLE(VECTOR_SIZE, stepElems) : stepElems;

                u8* in = (u8*)(p->input) + inOffset + (lineElem * INPUT_BPP);
                u8* out = (u8*)(p->output) + outOffset + (lineElem * INPUT_BPP);

                dma_start_2d(dma1, in, planeBuffer, numLines * stepElems * INPUT_BPP,
                              stepElems * INPUT_BPP, stepElems * INPUT_BPP,
                              p->istrides[0], numElems * INPUT_BPP);
                dma1.wait();

                op((half8*)planeBuffer, (half8*)planeBuffer, (half8*)p->weights, (half8*)p->biases, numLines, numElems, currentLine, p);

                dma_start_2d(dma1, planeBuffer, out, numLines * stepElems * INPUT_BPP,
                              stepElems * INPUT_BPP, stepElems * INPUT_BPP,
                              numElems * INPUT_BPP, p->ostrides[0]);
                dma1.wait();
            }

            subspace::incrementNCoord(setCoords, p->dims, p->ndims, numLines);
            currentLine += numLines;
        }

        nnLog(MVLOG_DEBUG, "%d # postOps_ND_core_2D : << sync\n", scGetShaveNumber());
    }
}

static uint8_t* cmxAlloc(PostOpsNDParams* params, int bytes)
{
    auto cmx = (u32)params->cmxSlice;
    auto aligned = (uint8_t*) ALIGN_TO_MULTIPLE(VECTOR_SIZE, cmx);

    params->cmxSlice = aligned + ALIGN_TO_MULTIPLE(VECTOR_SIZE, bytes);
    params->cmxSize -= ((u32)params->cmxSlice - cmx);

    return aligned;
};

static void allocAxisData(PostOpsNDParams* params)
{
    DmaAlShave dma;
    if (params->weights)
    {
        int bytes = params->axisSize * INPUT_BPP;
        fp16* weights = (fp16*) cmxAlloc(params, bytes);
        dma.start((u8*)params->weights, (u8*)weights, bytes);
        dma.wait();
        params->weights = weights;
    }
    if (params->biases)
    {
        int bytes = params->axisSize * INPUT_BPP;
        fp16* biases = (fp16*) cmxAlloc(params, bytes);
        dma.start((u8*)params->biases, (u8*)biases, bytes);
        dma.wait();
        params->biases = biases;
    }
}

static Operation selectOp(t_PostOps postOpType, int outerAxis)
{
    Operation op = nullptr;
    switch(postOpType)
    {
    case BIAS:
        op = outerAxis ? bias_fp16_outer : bias_fp16_inner;
        break;
    case BIAS_LEAKY_RELU:
        op = outerAxis ? bias_leaky_relu_fp16_outer : bias_leaky_relu_fp16_inner;
        break;
    case BIAS_RELU:
        op = outerAxis ? bias_relu_fp16_outer : bias_relu_fp16_inner;
        break;
    case PRELU:
        op = outerAxis ? prelu_fp16_outer : prelu_fp16_inner;
        break;
    case SCALE:
        op = outerAxis ? scale_fp16_outer : scale_fp16_inner;
        break;
    case SCALE_SHIFT:
        op = outerAxis ? scale_shift_fp16_outer : scale_shift_fp16_inner;
        break;

    case CLAMP:
        op = clamp_fp16;
        break;
    case ELU:
        op = elu_fp16;
        break;
    case LEAKY_RELU:
        op = leaky_relu_fp16;
        break;
    case POWER:
        op = power_fp16;
        break;
    case RELU:
        op = relu_fp16;
        break;
    case SIGMOID:
        op = sigmoid_fp16;
        break;
    case TANH:
        op = tanh_fp16;
        break;
    case LOG:
        op = log_fp16;
        break;
    case EXP:
        op = exp_fp16;
        break;
    case FLOOR:
        op = floor_fp16;
        break;
    case CEIL:
        op = ceil_fp16;
        break;
    case ROUND:
        op = round_fp16;
        break;
    case ERF:
        op = erf_fp16;
        break;
    case MISH:
        op = mish_fp16;
        break;
    case GELU:
        op = gelu_fp16;
        break;
    case SOFTPLUS:
        op = softplus_fp16;
        break;
    case SWISH:
        op = swish_fp16;
        break;
    case HSWISH:
        op = hswish_fp16;
        break;

    default:
        nnLog(MVLOG_DEBUG, "%2d # postOps_ND_core(): UNKNOWN <%d>\n", scGetShaveNumber(), (int)postOpType);
        break;
    }

    return op;
}

extern "C"
void postOps_ND_core(PostOpsNDParams* params)
{
    if (params->toProcess > 0) // extra SHAVEs have toProcess=0
    {
        const int outerAxis = (params->axis > 0) ? 1 : 0;
        Operation op = selectOp(params->postOpType, outerAxis);

        if (op)
        {
            Pipeline pipeline = params->useDma3D ? postOps_ND_core_3D : postOps_ND_core_2D;

            if (params->weights || params->biases)
                allocAxisData(params);

            pipeline(op, params);
        }
    }
}
