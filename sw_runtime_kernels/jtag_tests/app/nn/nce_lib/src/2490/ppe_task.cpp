/*
* {% copyright %}
*/
#include "ppe_task.h"
#include "nn_nce_lib_conversion_fbs.h"
#include "nn_nce_lib.h"
#include <nn_log.h>

namespace
{
    enum
    {
        PPE_READS_FROM_MPE = 0, //for convolution
        PPE_READS_FROM_MRM = 1, //for eltwise

        MPE0 = 0x10,
        MPE4 = 0x14,
    };
}

namespace nn
{
    namespace nce_lib
    {
        using namespace MVCNN;

        unsigned char ConfigFixedOpcode(const unsigned char opcode);
        unsigned char ConfigRsDtype(const MVCNN::DType dtype);
        unsigned char ConfigRdDtype(const MVCNN::DType dtype);

        bool DPUConfig::Setup_PPE(dpu_runtime::DPUInvariant& invariant)
        {
            auto in_tensor_ref = fb_invariant_->input_data();
            auto wt_tensor_ref = fb_invariant_->weights_data();
            auto out_tensor_ref = fb_invariant_->output_data();
            auto &registers = invariant.registers_;
            nnLog(MVLOG_DEBUG, "Setup PPETask at: %p", &registers);

            registers.ppe_scale.ppe_scale_bf.ppe_scale_round_shift = 1;
            registers.ppe_scale.ppe_scale_bf.ppe_scale_round = 1;
            registers.ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;

            auto post_shift = (out_tensor_ref->quant_post_shift_right()) ?
                out_tensor_ref->quant_post_shift_right() : 0;

            registers.ppe_remap_conv.ppe_remap_conv_bf.ppe_remap_shift_lr = post_shift < 0;
            registers.ppe_remap_conv.ppe_remap_conv_bf.ppe_remap_shift = abs(post_shift);


            auto out_zero_point = 0;
            if (opType_ != dpu_runtime::DPULayerTypes::DPU_MAXPOOL)
                if (const auto *quant_zero = out_tensor_ref->quant_zero())
                    out_zero_point = quant_zero->size() ? out_tensor_ref->quant_zero()->Get(0) : 0;

            switch (out_tensor_ref->data_dtype())
            {
                case DType_U8:
                    // U8 Quantization logic requires a final addition of the zero point
                    registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = out_zero_point;
                    break;

                case DType_I8:
                    if (out_zero_point != 0)
                        nnLog(MVLOG_ERROR, "Asymmetric I8 quantization is not supported by HW");
                    break;

                default:
                    break;
            }

            // Setting up configuration for single op activation
            unsigned char opcode = static_cast<uint8_t>(FixOpcodes::BYPASS);

            if (const auto *ppe_task = fb_invariant_->ppe_task())
            {
                if (const auto *instr_list = ppe_task->instruction_list_data())
                {
                    unsigned int count = instr_list->strides()->Get(STRIDES(B)) / sizeof(unsigned int);

                    if (count <= dpu_runtime::PPE_ILIST_ENTRIES * (sizeof(invariant.ppe_list_batch_size_) * 256 - 1))
                        invariant.ppe_list_batch_size_ = math::round_up<dpu_runtime::PPE_ILIST_ENTRIES>(count) / dpu_runtime::PPE_ILIST_ENTRIES;
                    else
                        nnLog(MVLOG_WARN, "PPE instruction list too long, containing %u instructions", count);
                }

                if (const auto *fixed_function = ppe_task->fixed_function())
                {
                    registers.ppe_scale_hclamp = fixed_function->Clamp_High();
                    registers.ppe_scale_lclamp = fixed_function->Clamp_Low();

                    if (const auto *ops = fixed_function->Ops())
                        if (ops->Length() > 0)
                        {
                            if (ops->Length() > 1)
                                nnLog(MVLOG_WARN, "%u PPE fixed opcodes specified. Using only the first one", ops->Length());

                            opcode = ConfigFixedOpcode(ops->Get(0));
                        }

                    switch (opcode)
                    {
                        case static_cast<unsigned char>(FixOpcodes::LPRELU):
                            registers.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = fixed_function->Lrelu_Mult();
                            registers.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = fixed_function->Lrelu_Shift();
                            break;
                    }
                }
            }

            nnLog(MVLOG_DEBUG, "PPE Opcode: 0x%02x", opcode);

            registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.opcode = opcode;
            registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rd_type = ConfigRdDtype(out_tensor_ref->data_dtype());
            registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rd = 0x10; //MEM


            // Post shift is not sign extending, avoid using U8 at output since the [0, 255] clamp
            // won't work with the fact that >> produces high positive numbers instead of negative
            if (post_shift > 0 && out_tensor_ref->data_dtype() == DType_U8)
            {
                registers.odu_cfg.odu_cfg_bf.dtype = ConfigOutputDtype(DType_I8);
                registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rd_type = ConfigRdDtype(DType_I8);
            }


            switch (opType_)
            {
                case dpu_runtime::DPULayerTypes::DPU_ELTWISE:
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.mrm_mode = PPE_READS_FROM_MRM;
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rs0_type = ConfigRsDtype(in_tensor_ref->data_dtype());
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rs0 = MPE0;
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rs1_type = ConfigRsDtype(wt_tensor_ref->data_dtype());
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rs1 = MPE4;

                    // Set PPE to read quant values from registers for eltwise since there are no weights tables
                    registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;

                    registers.ppe_scale.ppe_scale_bf.ppe_scale_mult =
                        (out_tensor_ref->quant_mult() && out_tensor_ref->quant_mult()->size()) ? out_tensor_ref->quant_mult()->Get(0) : 1;

                    registers.ppe_scale.ppe_scale_bf.ppe_scale_shift =
                        (out_tensor_ref->quant_shift() && out_tensor_ref->quant_shift()->size()) ? out_tensor_ref->quant_shift()->Get(0) : 0;

                    registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a =
                        (in_tensor_ref->quant_zero() && in_tensor_ref->quant_zero()->size()) ? in_tensor_ref->quant_zero()->Get(0) : 0;

                    registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b =
                        (wt_tensor_ref->quant_zero() && wt_tensor_ref->quant_zero()->size()) ? wt_tensor_ref->quant_zero()->Get(0) : 0;

                    break;

                default:
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.mrm_mode = PPE_READS_FROM_MPE;
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rs0_type = ConfigRsDtype(DType_I32);
                    registers.ppe_iram_fixed_instr.ppe_iram_fixed_instr_bf.rs0 = 0x04;//RR
                    if((in_tensor_ref->data_dtype() == DType_FP16) && (out_tensor_ref->data_dtype() == DType_U8))
                    {
                        nnLog(MVLOG_DEBUG, "Input FP16 and output U8 not supported for this op");
                        return false;
                    }

                    break;
            }
            return true;
        }

        unsigned char ConfigFixedOpcode(const unsigned char opcode)
        {
            switch (opcode)
            {
                case PPELayerType_STORE:   return static_cast<unsigned char>(FixOpcodes::STORE);
                case PPELayerType_LOAD:    return static_cast<unsigned char>(FixOpcodes::LOAD);
                case PPELayerType_CLEAR:   return static_cast<unsigned char>(FixOpcodes::CLEAR);
                case PPELayerType_NOOP:    return static_cast<unsigned char>(FixOpcodes::NOOP);
                case PPELayerType_HALT:    return static_cast<unsigned char>(FixOpcodes::HALT);
                case PPELayerType_ADD:     return static_cast<unsigned char>(FixOpcodes::ADD);
                case PPELayerType_SUB:     return static_cast<unsigned char>(FixOpcodes::SUB);
                case PPELayerType_MULT:    return static_cast<unsigned char>(FixOpcodes::MULT);
                case PPELayerType_LRELU:   return static_cast<unsigned char>(FixOpcodes::RELU);
                case PPELayerType_LRELUX:  return static_cast<unsigned char>(FixOpcodes::RELUX);
                case PPELayerType_LPRELU:  return static_cast<unsigned char>(FixOpcodes::LPRELU);
                case PPELayerType_MAXIMUM: return static_cast<unsigned char>(FixOpcodes::MAX);
                case PPELayerType_MINIMUM: return static_cast<unsigned char>(FixOpcodes::MIN);
                case PPELayerType_CEIL:    return static_cast<unsigned char>(FixOpcodes::CEIL);
                case PPELayerType_FLOOR:   return static_cast<unsigned char>(FixOpcodes::FLOOR);
                case PPELayerType_AND:     return static_cast<unsigned char>(FixOpcodes::AND);
                case PPELayerType_OR:      return static_cast<unsigned char>(FixOpcodes::OR);
                case PPELayerType_XOR:     return static_cast<unsigned char>(FixOpcodes::XOR);
                case PPELayerType_NOT:     return static_cast<unsigned char>(FixOpcodes::NOT);
                case PPELayerType_ABS:     return static_cast<unsigned char>(FixOpcodes::ABS);
                case PPELayerType_NEG:     return static_cast<unsigned char>(FixOpcodes::NEG);
                case PPELayerType_POW:     return static_cast<unsigned char>(FixOpcodes::POW);
                case PPELayerType_EXP:     return static_cast<unsigned char>(FixOpcodes::EXP);
                case PPELayerType_SIGMOID: return static_cast<unsigned char>(FixOpcodes::SIGMOID);
                case PPELayerType_TANH:    return static_cast<unsigned char>(FixOpcodes::TANH);
                case PPELayerType_SQRT:    return static_cast<unsigned char>(FixOpcodes::SQRT);
                case PPELayerType_RSQRT:   return static_cast<unsigned char>(FixOpcodes::RSQRT);
                case PPELayerType_FLEXARB: return static_cast<unsigned char>(FixOpcodes::FLEXARB);
                default:
                    nnLog(MVLOG_ERROR, "Invalid PPE opcode %u", opcode);
                    return static_cast<unsigned char>(FixOpcodes::INVALID_OPCODE);
            }
        }

        unsigned char ConfigRsDtype(const MVCNN::DType dtype)
        {
            switch (dtype)
            {
                case DType_FP16: return static_cast<unsigned char>(RsDtype::S1616);
                case DType_FP8:  return static_cast<unsigned char>(RsDtype::U8F);
                case DType_U8:   return static_cast<unsigned char>(RsDtype::G8);
                case DType_I32:  return static_cast<unsigned char>(RsDtype::I32);
                case DType_I8:   return static_cast<unsigned char>(RsDtype::I8);
                default:
                    nnLog(MVLOG_ERROR, "Invalid PPE RS datatype %u", dtype);
                    return static_cast<unsigned char>(RsDtype::INVALID_DTYPE);
            }
        }

        unsigned char ConfigRdDtype(const MVCNN::DType dtype)
        {
            switch (dtype)
            {
                case DType_FP16: return static_cast<unsigned char>(RdDtype::FP16);
                case DType_FP8:  return static_cast<unsigned char>(RdDtype::U8F);
                case DType_U8:   return static_cast<unsigned char>(RsDtype::G8);
                case DType_I32:  return static_cast<unsigned char>(RdDtype::I32);
                case DType_I8:   return static_cast<unsigned char>(RdDtype::I8);
                case DType_I4:   return static_cast<unsigned char>(RdDtype::I4);
                case DType_I2:   return static_cast<unsigned char>(RdDtype::I2);
                case DType_LOG:  return static_cast<unsigned char>(RdDtype::LOG);
                case DType_BIN:  return static_cast<unsigned char>(RdDtype::BIN);
                default:
                    nnLog(MVLOG_ERROR, "Invalid PPE RD datatype %u", dtype);
                    return static_cast<unsigned char>(RdDtype::INVALID_DTYPE);
            }
        }
    }
}
