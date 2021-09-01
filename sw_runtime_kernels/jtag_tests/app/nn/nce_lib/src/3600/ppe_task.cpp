/*
 * {% copyright %}
 */
#include "ppe_task.h"
#include "nn_nce_lib_conversion_fbs.h"
#include "nn_nce_lib.h"
#include <assert.h>
#include <nn_log.h>
#include <math.h>
#include <Fp16Convert.h>
#include <mv_types.h>

namespace nn {
namespace nce_lib {

#define PACK_B16(x, y, z)     ((x << 15) + (y <<  7) + (z))

enum {
    PPE_READS_FROM_MPE = 0, // for convolution
    PPE_READS_FROM_MRM = 1, // for eltwise

    MPE0 = 0x10,
    MPE4 = 0x14,
};

// PPE activation function choices in 2p7
enum activationFunction_t { no_activation_function, relu, relu_x, leaky_relu, unsupported };

struct activationFunctionDesc {
    float alpha;
    u32f32 alphaFP32;
    uint32_t alphaMult;  // Mult Register value
    uint32_t alphaShift; // Shift register value (number of bits to shift left by)
    activationFunction_t funcType;
    int32_t clampLow;
    int32_t clampHigh;

    activationFunctionDesc() :
        alpha (1.0),
        alphaMult (0),
        alphaShift (1),
        funcType (no_activation_function),
        clampLow(0),
        clampHigh(0)
        {
            alphaFP32.u32 = 0;
        }
};

// Constants for converting to BF16
constexpr uint32_t fp32FracBits = 23;
constexpr uint32_t fp16ExpBias = 15;
constexpr uint32_t fp16FracBits = 10;
constexpr uint32_t bf16FracBits = 7;
constexpr uint32_t bf16NanOutput = 0x7FC0;
constexpr uint32_t fp32NanOutput = 0x7FC00000; // Aligns with Synopsys DWC_FP_MULT fixed NAN output

using namespace MVCNN;

unsigned char ConfigRsDtype(const MVCNN::DType dtype);
unsigned char ConfigRdDtype(const MVCNN::DType dtype);

// Turn the integer mult and shift into a float (mult / 2**shift)
float integerPreluAlpha(uint32_t activationFunctionMult, uint32_t activationFunctionShift) {
    if (activationFunctionShift > 0)
        return (((float)activationFunctionMult) * (1.0 / pow(2, ((float)activationFunctionShift))));
    else
        return -1.0;
}

// Calculate the % difference between the calculated (HW/integer) alpha and our goal
float integerPreluAlphaDeltaPct(float targetPreluAlpha, float actualPreluAlpha) {
    if (abs(targetPreluAlpha) > 0)
        return abs(targetPreluAlpha - actualPreluAlpha) / abs(targetPreluAlpha);
    else
        return -1.0;
}

// Return -1 if actualAlpha < targetAlpha, 1 otherwise to determine approximation direction
int integerPreluAlphaDeltaSgn(float targetPreluAlpha, float actualPreluAlpha) {
    return (targetPreluAlpha >= actualPreluAlpha) ? 1 : -1;
}

// Approximate the HW integer prelu alpha settings given the target float alpha value in the blob
// Start at the largest values possible in the PPE registers and work backward until the target is reached
// If both fields reach 0 then we can't approximate this alpha value and return failure
bool approximatePreluAlpha(float targetAlpha, activationFunctionDesc &actFunctionDesc) {
    // Size of fields in PPE prelu register
    constexpr uint32_t intPreluMultBits = 11;
    constexpr uint32_t intPreluShiftBits = 5;

    int32_t mult = (1 << intPreluMultBits) - 1;
    int32_t shft = (1 << intPreluShiftBits) - 1;
    float approxAlpha = integerPreluAlpha(actFunctionDesc.alphaMult, actFunctionDesc.alphaShift);
    float alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
    int alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);
    float alphaErrorPctPrev = alphaErrorPct;
    int alphaErrorSgnPrev = alphaErrorSgn;

    bool multDescentDone = false;
    bool shftDescentDone = false;
    bool multDescentSuccess = false;
    bool shftDescentSuccess = false;

    // Decrease shift until the sign of the error changes
    while (!shftDescentDone && (shft > 0)) {
        shft--;
        approxAlpha = integerPreluAlpha(mult, shft);
        alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
        alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);

        // Error sign changed, we are as close as we can get with shift
        if (alphaErrorSgnPrev ^ alphaErrorSgn) {
            shftDescentSuccess = true;
            shftDescentDone = true;

            // Adjust for which approximation was closest to the actual alpha
            if (alphaErrorPctPrev < alphaErrorPct)
                shft++;

            approxAlpha = integerPreluAlpha(mult, shft);
        } else {
            alphaErrorPctPrev = alphaErrorPct;
            alphaErrorSgnPrev = alphaErrorSgn;
            if (shft == 0)
                shftDescentDone = true;
        }
    }

    // Decrease mult until the sign of the error changes
    if (shftDescentDone && shftDescentSuccess) {
        approxAlpha = integerPreluAlpha(mult, shft);
        alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
        alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);
        alphaErrorPctPrev = alphaErrorPct;
        alphaErrorSgnPrev = alphaErrorSgn;

        while (!multDescentDone && (mult > 0)) {
            mult--;
            approxAlpha = integerPreluAlpha(mult, shft);
            alphaErrorPct = integerPreluAlphaDeltaPct(targetAlpha, approxAlpha);
            alphaErrorSgn = integerPreluAlphaDeltaSgn(targetAlpha, approxAlpha);

            // Error sign changed, we are as close as we can get with mult
            if (alphaErrorSgnPrev ^ alphaErrorSgn) {
                multDescentSuccess = true;
                multDescentDone = true;

                // Adjust for which approximation was closest to the actual alpha
                if (alphaErrorPctPrev < alphaErrorPct)
                    mult++;

                approxAlpha = integerPreluAlpha(mult, shft);
            } else {
                alphaErrorPctPrev = alphaErrorPct;
                alphaErrorSgnPrev = alphaErrorSgn;
                multDescentDone = (mult == 0);
            }
        }

        // Found a solution
        if (multDescentDone && multDescentSuccess) {
            approxAlpha = integerPreluAlpha(mult, shft);
            actFunctionDesc.alphaMult = mult;
            actFunctionDesc.alphaShift = shft;

            mvLog(MVLOG_DEBUG,
                  "Approximating pReLU target alpha: %f actual alpha: %f ppe_prelu_mult: 0x%x ppe_prelu_shift: 0x%x",
                  targetAlpha, approxAlpha, actFunctionDesc.alphaMult, actFunctionDesc.alphaShift);
            return true;
        }
    }

    // Either mult or shift were 0 before approximation were complete
    mvLog(MVLOG_ERROR, "Failed to approximate pReLU target alpha: %f mult: 0x%x shft: 0x%x flags: %d, %d, %d %d",
          targetAlpha, mult, shft, shftDescentDone, shftDescentSuccess, multDescentDone, multDescentSuccess);
    return false;
}

// Apply RNE rounding to fp16 fractional part
// Note if overflow of frac (>10 bits) occurs when rounding up this will
// propagate to the exponent when added in the PACK16 function
uint32_t RoundFp16(uint32_t &dataIn, uint32_t fracWidth) {
    uint32_t frac;
    uint32_t precisionBitsMask;
    uint32_t precisionBits; // Bits used to determine precision
    uint32_t tie;

    if (fracWidth > fp16FracBits) {
        precisionBitsMask = (0x01 << (fracWidth - fp16FracBits)) - 1; // Bits to determine rounding
        precisionBits = dataIn & precisionBitsMask;
        frac = dataIn >> (fracWidth - fp16FracBits); // Pre-rounded fp16 fraction

        tie = 0x01 << (fracWidth - fp16FracBits - 1); // -1 so that we end up with leading 1-bit at MSB of precisionBits
        if (precisionBits > tie) {
            frac++;
        } else if (precisionBits == tie) {
            if ((frac & 0x01)) {
                frac++; // Add 1 if tie and frac is odd (ties to even)
            }
        }
    } else {
        precisionBits = 0;                           // No rounding needed
        frac = dataIn << (fp16FracBits - fracWidth); // fp16 fraction
    }

    return frac;
}

// Taken from ppeQuantisation.cpp in vpu_sysc:
//
// Convert the signed fixed point number to FP16
// Fixed point number is in 2's complement format so need to convert to ((-1)^S)*(1.x1x2...x9)*2^E format
// where: S is the sign
//        x1x2...x9 are the fractional bits after the leading 1-bit
//        E is the biased exponent
int32_t fixedPointToFp16(int32_t x, uint32_t intBits, uint32_t fracBits) {
    uint32_t result;
    uint32_t sign;
    int32_t exp;
    uint32_t frac;

    // Extract the sign and absolute value of x
    sign = (x >> (intBits + fracBits - 1)) & 0x01; // Extract sign bit (assumes signed fixed point input)
    uint32_t xAbs;
    if (sign) {
        xAbs = (~x + 1);
    } else {
        xAbs = x;
    }

    // Detect position of leading 1-bit of input (excluding the sign)
    uint32_t xAbsShift = xAbs;
    uint32_t count = 0;
    while (xAbsShift >>= 1) // Shift right until the leading 1 bit has been shifted off (xAbs becomes false)
    {
        count++;
    }

    // Calculate the fp16 exponent
    // (count - fracBits) is amount of bits shifted relative to fixed point decimal location
    exp = (int32_t(count) - int32_t(fracBits)) + int32_t(fp16ExpBias);

    // Calculate the fp16 fractional part (remaining bits after the leading 1-bit)
    uint32_t xAbsFrac;
    if (count == 0) // Input is zero or denorm
    {
        // Shift frac bits of fixed point input to fill upper bits of fp16 frac
        frac = xAbs << (fp16FracBits - count - 1);
    } else {
        xAbsFrac = xAbs ^ (0x01 << count); // Fractional part excluding leading 1-bit
        frac = RoundFp16(xAbsFrac, count);
    }

    result = (int32_t)PACK_F16(sign, exp, frac);

    return result;
}

// F32 to BFP16 conversion
unsigned int f32_to_b16_conv(unsigned int x, unsigned int rnd_mode, unsigned int *exceptions) {
    unsigned int result;
    unsigned int sign;
    int exp;
    unsigned int frac; //, res_frac;

    frac = EXTRACT_F32_FRAC(x);
    exp = EXTRACT_F32_EXP(x);
    sign = EXTRACT_F32_SIGN(x);

    if (exp == 0xFF) {
        // it's either a NaN or infinite
        if (frac != 0) {
            // NaN
            if (((frac >> 22) & 0x1) == 0x0) {
                // signalling NaN
                *exceptions |= F32_EX_INVALID;
            }
            result = 0x7FC0;
        } else {
            // infinity
            result = PACK_B16(sign, 0xFF, 0);
        }
    } else if (exp == 0x0) {
        if (frac != 0) {
            // Denormal
            // Flush to zero
            *exceptions |= (F32_EX_INEXACT | F32_EX_UNDERFLOW);
            result = PACK_B16(sign, 0, 0);
        } else {
            // Zero
            result = PACK_B16(sign, 0, 0);
        }
    } else {
        // Extract lsb, round and sticky bits
        int lsb = frac & 0x10000;
        int round = frac & 0x8000;
        int sticky = ((frac & 0x7fff) != 0) ? 1 : 0;

        // Truncate significand
        frac = frac >> 16;

        // Increment if necessary
        switch (rnd_mode) {
            case F32_RND_NEAREST_EVEN:
                if ((round && lsb) || (round && sticky))
                    frac = frac + 1;
                break;
            case F32_RND_TO_ZERO:
                break;
            case F32_RND_PLUS_INF:
                if ((sign == 0) && (round || sticky))
                    frac = frac + 1;
                break;
            case F32_RND_MINUS_INF:
                if ((sign == 1) && (round || sticky))
                    frac = frac + 1;
                break;
        }

        // Inexact if either round or sticky bit set
        if (round || sticky)
            *exceptions |= F32_EX_INEXACT;

        // Check if rounding caused significand overflow
        if ((frac & 0x80) != 0) {
            frac = 0;
            exp = exp + 1;
        }

        result = PACK_B16(sign, exp, frac);
    }

    return result;
}

bool areSupportedInputOutputTypes(MVCNN::DType in_type, MVCNN::DType out_type)
{
    bool in_supported = (in_type == DType_BFP16) || (in_type == DType_FP8) ||
          (in_type == DType_U8) || (in_type == DType_I8) ||
          (in_type == DType_FP16);

    bool out_supported = (out_type == DType_BFP16) || (out_type == DType_FP8) ||
          (out_type == DType_U8) || (out_type == DType_I8) ||
          (out_type == DType_FP16) || (out_type == DType_FP32);

    // Currently only support the following input & output types:
    if (!in_supported || !out_supported)
    {
        nnLog(MVLOG_ERROR, "Unsupported data type for PPE. In %s Out %s",
            EnumNameDType(in_type), EnumNameDType(out_type));
        return false;
    }

    return true;
}

// I8/U8 out
bool setupInt8(MVCNN::DType in_type, MVCNN::DType out_type, dpu_runtime::DPUInvariantRegisters &regs, activationFunctionDesc &actFuncDesc, uint8_t out_zero_point)
{
    if (in_type == DType_I8 || in_type == DType_U8)
    {
        // U8 in, INT32 convolution, I8/U8 out
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000; // INT32 convolution -> bypass FP clamp/gain
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
        regs.ppe_fp_prelu = 0;

        // Use integer scale table
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 0;
        regs.ppe_bias = 0; // Scale Table / Don't Care when ppe_scale_override is set
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
            0; // Scale Table / Don't Care when ppe_scale_override is set
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
            0; // Scale Table / Don't Care when ppe_scale_override is set
        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0; // As per above - not supporting override

        if (actFuncDesc.funcType == leaky_relu) {
            // in this case, we have to convert a high-precision floating-point
            // LeakyReLU alpha value to integer multiply and shift register values
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = actFuncDesc.alphaMult;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = actFuncDesc.alphaShift;
        } else if (actFuncDesc.funcType == relu_x) {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
        } else if (actFuncDesc.funcType == relu) {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
        } else {
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;  // no activation function
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0; // no activation function
        }
    } else {

        // FP16/BF16/FP8 in, FP32 convolution, U8 out
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
            0x004; // FP32 convolution -> INT32 (and eventually U8) out

        // Derive fp _prelu
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
        regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
        regs.ppe_bias = 0;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0; // can be overridden by LeakyReLU case

        // FP32 prelu
        if ((actFuncDesc.funcType == leaky_relu) || (actFuncDesc.funcType == relu) ||
            (actFuncDesc.funcType == relu_x)) {
            // for LeakyReLU, apply alpha; for ReLU and ReLUX, apply a negative-X slope of 0
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
            regs.ppe_fp_prelu = actFuncDesc.alphaFP32.u32;

            nnLog(MVLOG_DEBUG, "PPE_FIXED_FUNCTION,FP16PreluAlpha,%f,0x%08x", actFuncDesc.alphaFP32.f32,
                    regs.ppe_fp_prelu);
        }
    }

    //
    // U8 offset is added before clamping in VPU2.6
    regs.ppe_scale_lclamp = (out_type == DType_I8) ? -128 : 0;
    if (actFuncDesc.funcType == relu_x) {
        regs.ppe_scale_hclamp = (uint32_t)actFuncDesc.clampHigh;
    } else {
        regs.ppe_scale_hclamp = (out_type == DType_I8) ? 127 : 255;
    }

    // U8 Quantization logic requires a final addition of the zero point
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = out_zero_point;
    return true;
}

// FP8/FP16/FP32 out
bool setupFloat(MVCNN::DType in_type, MVCNN::DType out_type, dpu_runtime::DPUInvariantRegisters &regs, activationFunctionDesc &actFuncDesc)
{
    switch (in_type) {
        case DType_I8:
        case DType_U8: {
            if (out_type == DType_FP32) {
                nnLog(MVLOG_ERROR, "Input datatype %s with FP32 output is not supported",
                    EnumNameDType(in_type));
                return false;
            }
            // U8 in, INT32 convolution, FP16 out
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000; // INT32 convolution -> bypass FP clamp/gain
                                                                    //
            regs.ppe_misc.ppe_misc_bf.ppe_i32_convert =
                (out_type == DType_FP8)
                    ? 0x2
                    : 0x1; // INT32 s17.15 fixed-point convert to FP8/FP16

            // Use integer scale table
            regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 0; // Using integer scale table
            regs.ppe_bias = 0; // Scale Table / Don't Care when ppe_scale_override is set
            regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
                0; // Scale Table / Don't Care when ppe_scale_override is set
            regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
                0; // Scale Table / Don't Care when ppe_scale_override is set
            regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0; // As per above - not supporting override

            if (actFuncDesc.funcType == leaky_relu) {
                // in this case, we have to convert a high-precision floating-point
                // LeakyReLU alpha value to integer multiply and shift register values
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = actFuncDesc.alphaMult;
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = actFuncDesc.alphaShift;
            } else if (actFuncDesc.funcType == relu_x) {
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
            } else if (actFuncDesc.funcType == relu) {
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 0; // ReLU zero negative slope
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
            } else {
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;  // no activation function
                regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0; // no activation function
            }

            break;
        }
        case DType_BFP16:
        case DType_FP16:
        case DType_FP8: {
            // FP16 in, FP32 convolution, FP16 out
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
            if (out_type != DType_FP32)
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = (out_type == DType_FP8)
                                                                ? 0x003
                                                                : 0x001; // FP32 convolution -> FP8/FP16 out

            // FP32 Prelu
            if ((actFuncDesc.funcType == leaky_relu) || (actFuncDesc.funcType == relu) ||
                (actFuncDesc.funcType == relu_x)) {
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
                regs.ppe_fp_prelu =
                    actFuncDesc.alphaFP32.u32; // deliberately apply gain of zero to values less than zero
            } else {
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
            }

            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
            regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

            // Do not apply the scaling table to the integer PPE
            regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
            regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
            regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;

            break;
        }
        default:
        {
            nnLog(MVLOG_ERROR, "Support for input datatype %s with FP output is not yet implemented",
                    EnumNameDType(in_type));
            return false;
        }
    }

    // ReLUX is ReLU with an upper clamp
    if (actFuncDesc.funcType == relu_x) {
        uint32_t hclampAsFP16 = static_cast<uint32_t>(fixedPointToFp16((uint32_t)actFuncDesc.clampHigh, 32, 0));
        uint32_t hclampAsFP8 = ((hclampAsFP16 & 0x0000FF00) >> 8);

        // BF16 not yet supported here
        regs.ppe_scale_hclamp = (out_type == DType_FP8) ? hclampAsFP8 : hclampAsFP16;
        regs.ppe_scale_lclamp = 0x80000000;
    }
    else {
        // ReLU, LeakyReLU, unsupported
        regs.ppe_scale_hclamp = 0x7fffffff;
        regs.ppe_scale_lclamp = 0x80000000;
    }

    return true;
}

bool setupBFloat(MVCNN::DType in_dtype, MVCNN::DType out_type, dpu_runtime::DPUInvariantRegisters &regs, const activationFunctionDesc &actFunc)
{
    UNUSED(out_type);

    if (in_dtype == DType_I8 || in_dtype == DType_U8)
    {
        nnLog(MVLOG_ERROR, "X8 in, I32 convolution, BF16 out is not supported by the hardware");
        return false;
    }

    // FP8/FP16/BF16 in, FP32 convolution, BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x002; // FP32 convolution -> BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 1;     // Round to Nearest, Ties to Even (RNE)

    // FP32 Prelu
    if ((actFunc.funcType == leaky_relu) || (actFunc.funcType == relu) ||
        (actFunc.funcType == relu_x)) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
        regs.ppe_fp_prelu =
            actFunc.alphaFP32.u32; // deliberately apply gain of zero to values less than zero
    } else {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    }

    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;

    // Do not apply the scaling table to the integer PPE
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;

    // ReLUX is ReLU with an upper clamp
    if (actFunc.funcType == relu_x) {
        unsigned int exceptions = 0;
        u32f32 hClamp;
        hClamp.f32 = (float)((uint32_t)actFunc.clampHigh);

        regs.ppe_scale_hclamp = f32_to_b16_conv(hClamp.u32, F32_RND_NEAREST_EVEN, &exceptions);
        regs.ppe_scale_lclamp = 0x80000000;
    }
    else {
        // ReLU, LeakyReLU, unsupported
        regs.ppe_scale_hclamp = 0x7fffffff;
        regs.ppe_scale_lclamp = 0x80000000;
    }

    return true;
}

bool setupActivationFunction(const MVCNN::NCEInvariantFields *fb_invariant_,
                             const MVCNN::TensorReference *in_tensor_ref,
                             activationFunctionDesc &actFuncDesc)
{
    if (fb_invariant_->ppe_task() && fb_invariant_->ppe_task()->fixed_function()) {
        auto *ff = fb_invariant_->ppe_task()->fixed_function();

        if ((fb_invariant_->ppe_task()->fixed_function()->Ops()->size() > 0) &&
            ((uint32_t)(ff->Lrelu_Shift()) != 0x0)) {
            float lReluMult;
            int8_t lReluShift;

            // LeakyReLU: alpha slope derived according to Alessandro's Fathom test script as follows (ca. line 87:)
            // https://github.com/movidius/Fathom/blob/master/scripts/validation_test_script/mix_precision_blobs.py
            // scale_shift_to_fp(scale,shift): scale * 2 ** (-float(shift))
            // scale_shift_to_fp(ppe_ops["Lrelu_Mult"], ppe_ops["Lrelu_Shift"])
            lReluMult = (float)((uint32_t)ff->Lrelu_Mult());
            lReluShift = (int8_t)(((uint32_t)ff->Lrelu_Shift()) & 0xFF);

            auto rawAlpha = ldexp(lReluMult, -lReluShift);

            actFuncDesc.funcType = leaky_relu;
            actFuncDesc.alpha = (float)rawAlpha;

            if (in_tensor_ref->data_dtype() == DType_U8 || in_tensor_ref->data_dtype() == DType_I8) {
                auto apprxPreluAlphaSuccess =
                    approximatePreluAlpha(actFuncDesc.alpha, actFuncDesc);

                // approximatePreluAlpha already prints a warning if this fails, so just return failure
                if (!apprxPreluAlphaSuccess)
                    return false;
            }
        } else {
            if (((((uint32_t)ff->Clamp_High()) == 0x7FFFFFFF) || (((uint32_t)ff->Clamp_High()) == 0x00000000)) &&
                (((uint32_t)ff->Clamp_Low()) == 0x00000000)) {
                // ReLU
                actFuncDesc.funcType = relu;
                actFuncDesc.alpha = -0.0; // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                          // (0x00000000), not negative zero (0x80000000)
            } else if ((((uint32_t)ff->Clamp_High()) < 0x7FFFFFFF) && (((uint32_t)ff->Clamp_Low()) == 0x00000000)) {
                // ReLUX
                actFuncDesc.funcType = relu_x;
                actFuncDesc.alpha = -0.0; // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                          // (0x00000000), not negative zero (0x80000000)
            } else {
                // No activation function, or unrecognised activation function
                actFuncDesc.funcType = no_activation_function;
            }
        }

        actFuncDesc.alphaFP32.f32 = actFuncDesc.alpha; // alpha (accessible as uint32_t FP32 bit pattern in .u)

        actFuncDesc.clampHigh = ff->Clamp_High();
        actFuncDesc.clampLow = ff->Clamp_Low();
    }

    return true;
}

bool DPUConfig::Setup_PPE(dpu_runtime::DPUInvariant &invariant) {
    auto in_tensor_ref = fb_invariant_->input_data();
    auto wt_tensor_ref = fb_invariant_->weights_data();
    auto out_tensor_ref = fb_invariant_->output_data();

    auto &regs = invariant.registers_;

    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0; // Eltwise uses this as we don't have a weights table to source bias from
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0; // Eltwise uses this as we don't have a weights table to source bias from
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = 0; // Used to set the zero point for u8

    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 0; // No short term plan to support this
    regs.ppe_bias = 0;                                            // As per above - not supporting override
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;               // As per above - not supporting override
    regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0;              // As per above - not supporting override
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;              // As per above - not supporting override

    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0; // No short term plan to support this
    regs.ppe_fp_bias = 0;                                            // As per above - not supporting override
    regs.ppe_fp_scale = 0;                                           // As per above - not supporting override

    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;  // Serialised in fixed function
    regs.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0; // Serialised in fixed function
    regs.ppe_scale_hclamp = 0; // Default values applied per output data type, may be serialised in ff
    regs.ppe_scale_lclamp = 0; // Default values applied per output data type, may be serialised in ff
    regs.ppe_misc.ppe_misc_bf.ppe_i32_convert = 0; // Use in mixed precision when going fixed -> float point, infer
    regs.ppe_misc.ppe_misc_bf.ppe_fp16_clamp = 0;  // Not serialised
    regs.ppe_misc.ppe_misc_bf.ppe_fp16_ftz = 0;    // Not serialised
    regs.ppe_fp_prelu = 1;                         // Derive from ppe_prelu_mult & ppe_prelu_shift
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;  // Set based on data types, if we see float point - don't bypass!
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0; // Default to no conversion but set based on output data type

    // Max pool has no U8 zero subtraction hence
    uint8_t out_zero_point =
        (out_tensor_ref->quant_zero()->size() && fb_invariant_->dpu_task_type() != DPULayerType_MAXPOOL)
            ? out_tensor_ref->quant_zero()->Get(0)
            : 0;

    activationFunctionDesc actFuncDesc;

    if (!areSupportedInputOutputTypes(in_tensor_ref->data_dtype(), out_tensor_ref->data_dtype()))
        return false;

    // Check if there's a PPE fixed-function task, and if so, whether it
    // could be an activation function (requires GFS 3.22.x)
    if (!setupActivationFunction(fb_invariant_, in_tensor_ref, actFuncDesc))
        return false;

    bool successful = false;

    switch (out_tensor_ref->data_dtype()) {
        case DType_I8:
        case DType_U8:
            successful = setupInt8(in_tensor_ref->data_dtype(), out_tensor_ref->data_dtype(), regs, actFuncDesc, out_zero_point);
            break;
        case DType_FP8:
        case DType_FP16:
        case DType_FP32:
            successful = setupFloat(in_tensor_ref->data_dtype(), out_tensor_ref->data_dtype(), regs, actFuncDesc);
            break;
        case DType_BFP16:
            successful = setupBFloat(in_tensor_ref->data_dtype(), out_tensor_ref->data_dtype(), regs, actFuncDesc);
            break;
        default:
            nnLog(MVLOG_ERROR, "only U8, I8, FP16 are currently supported for BF16 out");
            successful = false;
    }

    if (!successful)
        return false;

    if (fb_invariant_->dpu_task_type() == DPULayerType_ELTWISE) {
        // Set PPE to read quant values from registers for eltwise since there
        // are no weights tables
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = fb_invariant_->ppe_task()->rounding();

        // For supporting the MTL-style scales case, mult/shift will need to be adjusted along with the
        // code in eltwise.cpp
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
            out_tensor_ref->quant_mult()->size() ? out_tensor_ref->quant_mult()->Get(0) : 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
            out_tensor_ref->quant_shift()->size() ? out_tensor_ref->quant_shift()->Get(0) : 0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a =
            in_tensor_ref->quant_zero()->size() ? in_tensor_ref->quant_zero()->Get(0) : 0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b =
            wt_tensor_ref->quant_zero()->size() ? wt_tensor_ref->quant_zero()->Get(0) : 0;
    } else if (fb_invariant_->dpu_task_type() == DPULayerType_MAXPOOL) {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0x3; // 0x3 - no round
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 0x1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0x0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0x0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0x0;

        if (in_tensor_ref->data_dtype() == DType_FP16) {
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0x1;
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                0x0; // FP16 MaxPool result is already FP16 with CRL FP MAC => no conversion
            // regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override  = 1; // No short term plan to support this
            // regs.ppe_fp_scale           = 0x3f800000; // FP32 1.0x
            // regs.ppe_fp_bias            = 0x0;
        }
    }

    return true;
}

unsigned char ConfigRsDtype(const MVCNN::DType dtype) {
    switch (dtype) {
        case DType_FP16:
            return static_cast<unsigned char>(RsDtype::S1616);
        case DType_FP8:
            return static_cast<unsigned char>(RsDtype::U8F);
        case DType_U8:
            return static_cast<unsigned char>(RsDtype::G8);
        case DType_I32:
            return static_cast<unsigned char>(RsDtype::I32);
        case DType_I8:
            return static_cast<unsigned char>(RsDtype::I8);
        default:
            nnLog(MVLOG_ERROR, "Invalid PPE RS datatype %u", dtype);
            return static_cast<unsigned char>(RsDtype::INVALID_DTYPE);
    }
}

unsigned char ConfigRdDtype(const MVCNN::DType dtype) {
    switch (dtype) {
        case DType_FP16:
            return static_cast<unsigned char>(RdDtype::FP16);
        case DType_FP8:
            return static_cast<unsigned char>(RdDtype::U8F);
        case DType_U8:
            return static_cast<unsigned char>(RsDtype::G8);
        case DType_I32:
            return static_cast<unsigned char>(RdDtype::I32);
        case DType_I8:
            return static_cast<unsigned char>(RdDtype::I8);
        case DType_I4:
            return static_cast<unsigned char>(RdDtype::I4);
        case DType_I2:
            return static_cast<unsigned char>(RdDtype::I2);
        case DType_LOG:
            return static_cast<unsigned char>(RdDtype::LOG);
        case DType_BIN:
            return static_cast<unsigned char>(RdDtype::BIN);
        default:
            nnLog(MVLOG_ERROR, "Invalid PPE RD datatype %u", dtype);
            return static_cast<unsigned char>(RdDtype::INVALID_DTYPE);
    }
}
} // namespace nce_lib
} // namespace nn
