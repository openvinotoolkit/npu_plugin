/*
 * {% copyright %}
 */
#include <convert.h>
#include <assert.h>
#include <nn_log.h>
#include <math.h>
#include <fp_utils.h>
#include <mv_types.h>
#include <nn_logging.h>

namespace parsing_lib {

enum {
    PPE_READS_FROM_MPE = 0, // for convolution
    PPE_READS_FROM_MRM = 1, // for eltwise

    MPE0 = 0x10,
    MPE4 = 0x14,
};

enum class RsDtype : unsigned char { FP16, U8F, G8, I8, I32, S1616, INVALID_DTYPE };
enum class RdDtype : unsigned char { FP16, U8F, G8, I8, I32, I4, I2, LOG, BIN, INVALID_DTYPE };

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

    activationFunctionDesc()
        : alpha(1.0)
        , alphaMult(0)
        , alphaShift(1)
        , funcType(no_activation_function)
        , clampLow(0)
        , clampHigh(0) {
        alphaFP32.u32 = 0;
    }
};

unsigned char ConfigRsDtype(DType dtype);
unsigned char ConfigRdDtype(DType dtype);

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

bool areSupportedInputOutputTypes(DType in_type, DType out_type) {
    bool in_supported = (in_type == DType::BFP16) || (in_type == DType::FP8) || (in_type == DType::U8) ||
                        (in_type == DType::I8) || (in_type == DType::I4) || (in_type == DType::FP16);

    bool out_supported = (out_type == DType::BFP16) || (out_type == DType::FP8) || (out_type == DType::U8) ||
                         (out_type == DType::I8) || (out_type == DType::I32) || (out_type == DType::I4) ||
                         (out_type == DType::FP16) || (out_type == DType::FP32);

    // Currently only support the following input & output types:
    if (!in_supported || !out_supported) {
        nnLog(MVLOG_ERROR, "Unsupported data type for PPE. In %d Out %d", to_underlying(in_type),
              to_underlying(out_type));
        return false;
    }

    return true;
}

bool setupInt(DType in_type, DType out_type, DPUInvariantRegisters &regs,
               activationFunctionDesc &actFuncDesc, uint8_t out_zero_point) {
    if (in_type == DType::I8 || in_type == DType::U8 || in_type == DType::I4) {
        // I8/U8/I4 in, INT32 convolution, I8/U8/I4 out
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000; // INT32 convolution -> bypass FP clamp/gain
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
        regs.ppe_fp_prelu = 0;

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
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x004; // FP32 convolution -> INT32 (and eventually U8) out

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
    switch (out_type) {
        case DType::I4:
            regs.ppe_scale_lclamp = -8;
            break;
        case DType::I8:
            regs.ppe_scale_lclamp = -128;
            break;
        case DType::U8:
            regs.ppe_scale_lclamp = 0;
            break;
        case DType::I32:
            regs.ppe_scale_lclamp = 0x80000000;
            break;
        default:
            nnLog(MVLOG_DEBUG, "Unexpected dtype: %d", to_underlying(out_type));
            return false;
    }
    if (actFuncDesc.funcType == relu_x) {
        regs.ppe_scale_hclamp = (uint32_t)actFuncDesc.clampHigh;
    } else {
        switch (out_type) {
            case DType::I4:
                regs.ppe_scale_hclamp = 7;
                break;
            case DType::I8:
                regs.ppe_scale_hclamp = 127;
                break;
            case DType::U8:
                regs.ppe_scale_hclamp = 255;
                break;
            case DType::I32:
                regs.ppe_scale_hclamp = 0x7FFFFFFF;
                break;
            default:
                nnLog(MVLOG_DEBUG, "Unexpected dtype: %d", to_underlying(out_type));
                return false;
        }
    }

    // U8 Quantization logic requires a final addition of the zero point
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = out_zero_point;
    return true;
}

// FP8/FP16/FP32 out
bool setupFloat(DType in_type, DType out_type, DPUInvariantRegisters &regs,
                activationFunctionDesc &actFuncDesc) {
    switch (in_type) {
        case DType::I8:
        case DType::U8: {
            if (out_type == DType::FP32) {
                nnLog(MVLOG_ERROR, "Input datatype %d with FP32 output is not supported", to_underlying(in_type));
                return false;
            }
            // U8 in, INT32 convolution, FP16 out
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 1;
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x000; // INT32 convolution -> bypass FP clamp/gain
                                                                  //
            regs.ppe_misc.ppe_misc_bf.ppe_i32_convert =
                (out_type == DType::FP8) ? 0x2 : 0x1; // INT32 s17.15 fixed-point convert to FP8/FP16

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
        case DType::BFP16:
        case DType::FP16:
        case DType::FP8: {
            // FP16 in, FP32 convolution, FP16 out
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
            if (out_type != DType::FP32)
                regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                    (out_type == DType::FP8) ? 0x003 : 0x001; // FP32 convolution -> FP8/FP16 out

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
        default: {
            nnLog(MVLOG_ERROR, "Support for input datatype %d with FP output is not yet implemented",
                  to_underlying(in_type));
            return false;
        }
    }

    // ReLUX is ReLU with an upper clamp
    if (actFuncDesc.funcType == relu_x) {
        uint32_t hclampAsFP16 = static_cast<uint32_t>(fixedPointToFp16((uint32_t)actFuncDesc.clampHigh, 32, 0));
        uint32_t hclampAsFP8 = ((hclampAsFP16 & 0x0000FF00) >> 8);

        // BF16 not yet supported here
        regs.ppe_scale_hclamp = (out_type == DType::FP8) ? hclampAsFP8 : hclampAsFP16;
        regs.ppe_scale_lclamp = 0x80000000;
    } else {
        // ReLU, LeakyReLU, unsupported
        regs.ppe_scale_hclamp = 0x7fffffff;
        regs.ppe_scale_lclamp = 0x80000000;
    }

    return true;
}

bool setupBFloat(DType in_dtype, DType out_type, DPUInvariantRegisters &regs,
                 const activationFunctionDesc &actFunc) {
    UNUSED(out_type);

    if (in_dtype == DType::I8 || in_dtype == DType::U8) {
        nnLog(MVLOG_ERROR, "X8 in, I32 convolution, BF16 out is not supported by the hardware");
        return false;
    }

    // FP8/FP16/BF16 in, FP32 convolution, BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 0x002; // FP32 convolution -> BF16 out
    regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 1;     // Round to Nearest, Ties to Even (RNE)

    // FP32 Prelu
    if ((actFunc.funcType == leaky_relu) || (actFunc.funcType == relu) || (actFunc.funcType == relu_x)) {
        regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 1;
        regs.ppe_fp_prelu = actFunc.alphaFP32.u32; // deliberately apply gain of zero to values less than zero
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
    } else {
        // ReLU, LeakyReLU, unsupported
        regs.ppe_scale_hclamp = 0x7fffffff;
        regs.ppe_scale_lclamp = 0x80000000;
    }

    return true;
}

bool setupActivationFunction(const Invariant &srcInvariant,
                             const Optional<TensorReference> &in_tensor_ref, activationFunctionDesc &actFuncDesc) {
    auto &ff = srcInvariant.ppe_task.fixed_function;

    // What is the point of having one op? just to decided LRELU vs others?
    if ((srcInvariant.ppe_task.fixed_function.Ops.size() > 0) &&
        ((uint32_t)(ff.Lrelu_Shift) != 0x0)) {
        float lReluMult;
        int8_t lReluShift;

        // LeakyReLU: alpha slope derived according to Alessandro's Fathom test script as follows (ca. line 87:)
        // https://github.com/movidius/Fathom/blob/master/scripts/validation_test_script/mix_precision_blobs.py
        // scale_shift_to_fp(scale,shift): scale * 2 ** (-float(shift))
        // scale_shift_to_fp(ppe_ops["Lrelu_Mult"], ppe_ops["Lrelu_Shift"])
        lReluMult = (float)((uint32_t)ff.Lrelu_Mult);
        lReluShift = (int8_t)(((uint32_t)ff.Lrelu_Shift) & 0xFF);

        auto rawAlpha = ldexp(lReluMult, -lReluShift);

        actFuncDesc.funcType = leaky_relu;
        actFuncDesc.alpha = (float)rawAlpha;

        if (in_tensor_ref->data_dtype == DType::U8 || in_tensor_ref->data_dtype == DType::I8 ||
            in_tensor_ref->data_dtype == DType::I32) {
            auto apprxPreluAlphaSuccess = approximatePreluAlpha(actFuncDesc.alpha, actFuncDesc);

            // approximatePreluAlpha already prints a warning if this fails, so just return failure
            if (!apprxPreluAlphaSuccess)
                return false;
        }
    } else {
        if (((((uint32_t)ff.Clamp_High) == 0x7FFFFFFF) || (((uint32_t)ff.Clamp_High) == 0x00000000)) &&
            (((uint32_t)ff.Clamp_Low) == 0x00000000)) {
            // ReLU
            actFuncDesc.funcType = relu;
            actFuncDesc.alpha = -0.0; // note: -0.0, to ensure zero-gained data uses positive zero in FP32
                                        // (0x00000000), not negative zero (0x80000000)
        } else if ((((uint32_t)ff.Clamp_High) < 0x7FFFFFFF) && (((uint32_t)ff.Clamp_Low) == 0x00000000)) {
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

    actFuncDesc.clampHigh = ff.Clamp_High;
    actFuncDesc.clampLow = ff.Clamp_Low;

    return true;
}

void setupTaskType(const Invariant &srcInvariant, DPUInvariant &invariant) {
    auto &regs = invariant.registers;
    auto &in_tensor_ref = srcInvariant.input_data;
    auto &wt_tensor_ref = srcInvariant.weights_data;
    auto &out_tensor_ref = srcInvariant.output_data;

    const bool isFP16 = in_tensor_ref->data_dtype == DType::FP16 || in_tensor_ref->data_dtype == DType::BFP16;
    const bool isFP = isFP16 || in_tensor_ref->data_dtype == DType::FP32 || in_tensor_ref->data_dtype == DType::FP8;
    if (srcInvariant.dpu_task_type == DPULayerType::ELTWISE) {
        // Set PPE to read quant values from registers for eltwise since there
        // are no weights tables
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = to_underlying(srcInvariant.ppe_task.rounding);

        // For supporting the MTL-style scales case, mult/shift will need to be adjusted along with the
        // code in eltwise.cpp
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
            out_tensor_ref->quant_mult.size() ? out_tensor_ref->quant_mult[0] : 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
            out_tensor_ref->quant_shift.size() ? out_tensor_ref->quant_shift[0] : 0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a =
            in_tensor_ref->quant_zero.size() ? in_tensor_ref->quant_zero[0] : 0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b =
            wt_tensor_ref->quant_zero.size() ? wt_tensor_ref->quant_zero[0] : 0;

        if (isFP) {
            regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0x1;
            regs.ppe_fp_scale = 0x3f800000; // fp32 equiv of 1
            regs.ppe_fp_bias = 0x0;
        }
    } else if (srcInvariant.dpu_task_type == DPULayerType::MAXPOOL) {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0x3; // 0x3 - no round
        regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 0x1;
        regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0x0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0x0;
        regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0x0;

        if (isFP16) {
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0x1;
            regs.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert =
                0x0; // FP16 MaxPool result is already FP16 with CRL FP MAC => no conversion
        }
        if (isFP) {
            regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0x1;
            regs.ppe_fp_scale = 0x3f800000; // fp32 equiv of 1
            regs.ppe_fp_bias = 0x0;
        }
    } else if (regs.elops_wload.elops_wload_bf.pool_wt_rd_dis &&
               srcInvariant.dpu_task_type == DPULayerType::AVEPOOL) {
        regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
        u32f32 fp32_scale;
        fp32_scale.f32 = 1.0f / (srcInvariant.kernelH * srcInvariant.kernelW);
        switch (wt_tensor_ref->data_dtype) {
            case DType::I8:
            case DType::U8:
                regs.ppe_scale.ppe_scale_bf.ppe_scale_mult =
                    wt_tensor_ref->quant_mult.size() ? wt_tensor_ref->quant_mult[0] : 1;
                regs.ppe_scale.ppe_scale_bf.ppe_scale_shift =
                    wt_tensor_ref->quant_shift.size() ? wt_tensor_ref->quant_shift[0] : 0;
                break;
            case DType::FP16:
            case DType::BFP16:
                regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 1;
                regs.ppe_fp_scale = fp32_scale.u32;
                break;
            default:
                break;
        }
    }
}

bool DPUConfigurator::Setup_PPE(DPUInvariant &invariant) {
    auto &in_tensor_ref = srcInvariant.input_data;
    auto &out_tensor_ref = srcInvariant.output_data;

    auto &regs = invariant.registers;

    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0; // Eltwise uses this as we don't have a weights table to source bias from
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0; // Eltwise uses this as we don't have a weights table to source bias from
    regs.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = 0; // Used to set the zero point for u8

    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 0;
    regs.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0;
    regs.ppe_bias = 0;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_round = 0;
    regs.ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
    regs.ppe_fp_bias = 0;
    regs.ppe_fp_scale = 0;

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
        (out_tensor_ref->quant_zero.size() && srcInvariant.dpu_task_type != DPULayerType::MAXPOOL)
            ? out_tensor_ref->quant_zero[0]
            : 0;

    activationFunctionDesc actFuncDesc;

    if (!areSupportedInputOutputTypes(in_tensor_ref->data_dtype, out_tensor_ref->data_dtype))
        return false;

    // Check if there's a PPE fixed-function task, and if so, whether it
    // could be an activation function (requires GFS 3.22.x)
    if (!setupActivationFunction(srcInvariant, in_tensor_ref, actFuncDesc))
        return false;

    bool successful = false;

    switch (out_tensor_ref->data_dtype) {
        case DType::I4:
        case DType::I8:
        case DType::U8:
        case DType::I32:
            successful =
                setupInt(in_tensor_ref->data_dtype, out_tensor_ref->data_dtype, regs, actFuncDesc, out_zero_point);
            break;
        case DType::FP8:
        case DType::FP16:
        case DType::FP32:
            successful = setupFloat(in_tensor_ref->data_dtype, out_tensor_ref->data_dtype, regs, actFuncDesc);
            break;
        case DType::BFP16:
            successful = setupBFloat(in_tensor_ref->data_dtype, out_tensor_ref->data_dtype, regs, actFuncDesc);
            break;
        default:
            nnLog(MVLOG_ERROR, "only U8, I8, FP16 are currently supported for BF16 out");
            successful = false;
    }

    if (!successful)
        return false;

    setupTaskType(srcInvariant, invariant);

    return true;
}

unsigned char ConfigRsDtype(DType dtype) {
    switch (dtype) {
        case DType::FP16:
            return static_cast<unsigned char>(RsDtype::S1616);
        case DType::FP8:
            return static_cast<unsigned char>(RsDtype::U8F);
        case DType::U8:
            return static_cast<unsigned char>(RsDtype::G8);
        case DType::I32:
            return static_cast<unsigned char>(RsDtype::I32);
        case DType::I8:
            return static_cast<unsigned char>(RsDtype::I8);
        default:
            nnLog(MVLOG_ERROR, "Invalid PPE RS datatype %u", dtype);
            return static_cast<unsigned char>(RsDtype::INVALID_DTYPE);
    }
}

unsigned char ConfigRdDtype(DType dtype) {
    switch (dtype) {
        case DType::FP16:
            return static_cast<unsigned char>(RdDtype::FP16);
        case DType::FP8:
            return static_cast<unsigned char>(RdDtype::U8F);
        case DType::U8:
            return static_cast<unsigned char>(RsDtype::G8);
        case DType::I32:
            return static_cast<unsigned char>(RdDtype::I32);
        case DType::I8:
            return static_cast<unsigned char>(RdDtype::I8);
        case DType::I4:
            return static_cast<unsigned char>(RdDtype::I4);
        case DType::I2:
            return static_cast<unsigned char>(RdDtype::I2);
        case DType::LOG:
            return static_cast<unsigned char>(RdDtype::LOG);
        case DType::BIN:
            return static_cast<unsigned char>(RdDtype::BIN);
        default:
            nnLog(MVLOG_ERROR, "Invalid PPE RD datatype %u", dtype);
            return static_cast<unsigned char>(RdDtype::INVALID_DTYPE);
    }
}
} // namespace parsing_lib
