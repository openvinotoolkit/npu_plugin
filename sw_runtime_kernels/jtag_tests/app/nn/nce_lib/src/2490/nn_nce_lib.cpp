/*
* {% copyright %}
*/
#include "nn_nce_lib.h"
#include "nn_nce_lib_utils.h"
#include "nn_nce_lib_conversion_fbs.h"
#include "ppe_task.h"
#include <nn_math.h>
#include <string.h>
#include <limits.h>

#include <nn_inference_runtime_types.h>
#include <nn_log.h>

#define WORKAROUND_FORCE_WEIGHTS_OFFSET_TO_NNCMX_BASE_ADR

#if defined(CONFIG_TARGET_SOC_MA2490) && defined(REVISION_A0)
#define NN_WORKAROUND_33243_A0
#endif

using namespace nn::dpu_runtime;

namespace
{
    enum
    {
        KERNEL_SIZE_MIN = 1,
        KERNEL_SIZE_MAX = 11,
        KERNEL_STRIDE_MIN = 1,
        KERNEL_STRIDE_MAX = 8,
    };

    DPULayerTypes mapOpType(MVCNN::DPULayerType fb_op)
    {
        using namespace MVCNN;

        switch (fb_op)
        {
            case MVCNN::DPULayerType_CONV:    return DPU_CONV;
            case MVCNN::DPULayerType_DWCONV:  return DPU_DWCONV;
            case MVCNN::DPULayerType_MAXPOOL: return DPU_MAXPOOL;
            case MVCNN::DPULayerType_AVEPOOL: return DPU_AVEPOOL;
            case MVCNN::DPULayerType_ELTWISE: return DPU_ELTWISE;
            case MVCNN::DPULayerType_CMCONV:  return DPU_CMCONV;
            default:
                nnLog(MVLOG_ERROR, "DPU op type %u not supported", fb_op);
                return NO_OP;
        }
    }
}

namespace nn
{
    namespace nce_lib
    {
        DPUConfig::DPUConfig(const MVCNN::NCEInvariantFields *fb_invariant, const MVCNN::NCEVariantFields *fb_variant, unsigned int cluster_count) :
            fb_invariant_(fb_invariant),
            fb_variant_(fb_variant),
            opType_(fb_invariant_ ? mapOpType(fb_invariant_->dpu_task_type()) : NO_OP),
            cluster_count_(cluster_count)
        {
        }

        bool DPUConfig::workaround_for_bug_33243() const
        {
#ifdef NN_WORKAROUND_33243_A0
            return (opType_ == DPU_DWCONV || opType_ == DPU_MAXPOOL) &&
                (fb_invariant_->kernel_padLeft() & 1) &&
                fb_invariant_->is_segmented();
#else
            return false;
#endif
        }

        bool DPUConfig::Setup_Invariant(DPULayerTypes &opType, DPUInvariant& invariant, DPUAddresses &addresses)
        {
            if (opType_ == NO_OP)
                return false;

            if(fb_invariant_->weights_data() &&
              (!Is_Dtype_Mix_Supported(fb_invariant_->input_data()->data_dtype(), fb_invariant_->weights_data()->data_dtype())))
                return false;

            nnLog(MVLOG_DEBUG, "Layer type is %u", opType_);
            opType = opType_;

            DPUInvariantRegisters& registers_ = invariant.registers_;
            memset(&registers_, 0, sizeof(registers_));

            if (!Setup_Input(registers_))
                return false;

            if (!Setup_Weights(registers_))
                return false;

            if (!Setup_Kernel(registers_))
                return false;

            if (!Setup_Output(registers_))
                return false;

            switch (opType_)
            {
                case nn::dpu_runtime::DPU_CONV:
                    SetupInvariant_Convolution(registers_);
                    break;

                case nn::dpu_runtime::DPU_CMCONV:
                    if(!SetupInvariant_CMConv(registers_)) return false;

                    // Pre-compute the stride for CM conv to reduce SNN code size
                    // Each line and plane has to be aligned to 16 bytes
                    invariant.channel_major_stride_ =
                        registers_.tensor_size0.tensor_size0_bf.tensor_size_y *
                        (nn::math::round_up<16 * 8>(registers_.tensor_size0.tensor_size0_bf.tensor_size_x *
                        getIDUDTypeSizeBits(static_cast<InputTensorDType>(registers_.tensor_mode.tensor_mode_bf.amode))) >> 3);
                    break;

                case nn::dpu_runtime::DPU_MAXPOOL:
                    SetupInvariant_MaxPool(registers_);
                    break;

                case nn::dpu_runtime::DPU_AVEPOOL:
                case nn::dpu_runtime::DPU_DWCONV:
                    SetupInvariant_DwConvolution(registers_);
                    break;

                case nn::dpu_runtime::DPU_ELTWISE:
                    if (!SetupInvariant_Eltwise(registers_)) return false;
                    break;

                default:
                    nnLog(MVLOG_ERROR, "DPU op type not supported");
                    return false;
            }

            if(!SetupInvariant_RelativeAddresses(addresses)) return false;
            if(!Setup_PPE(invariant)) return false;

            if(!apply_workaround_for_bug_33243(registers_)) return false;

            invariant.cluster_ = math::firstBitIndex(addresses.input_.index());

            return true;
        }

        bool DPUConfig::Setup_Variant(DPUVariant& variant, DPUInvariant& invariant)
        {
            memset(reinterpret_cast<void*>(&variant.registers_), 0, sizeof(nn::dpu_runtime::DPUVariantRegisters));

            auto stride_w = fb_invariant_->kernel_strideW();
            auto stride_h = fb_invariant_->kernel_strideH();
            auto K_w = fb_invariant_->kernelW();
            auto K_h = fb_invariant_->kernelH();

            auto global_PT = fb_invariant_->kernel_padTop();
            auto global_PL = fb_invariant_->kernel_padLeft();

            auto local_PT = fb_variant_->padTop();
            auto local_PB = fb_variant_->padBottom();
            auto local_PL = fb_variant_->padLeft();
            auto local_PR = fb_variant_->padRight();

            auto output_start_x = fb_variant_->workload_start_X();
            auto output_start_y = fb_variant_->workload_start_Y();
            auto output_start_z = fb_variant_->workload_start_Z();

            auto output_end_x = fb_variant_->workload_end_X();
            auto output_end_y = fb_variant_->workload_end_Y();
            auto output_end_z = fb_variant_->workload_end_Z();

            auto op_size_x = output_end_x - output_start_x + 1;
            auto op_size_y = output_end_y - output_start_y + 1;
            auto op_size_z = output_end_z - output_start_z + 1;

            variant.registers_.workload_size0.workload_size0_bf.workload_size_x = stride_w * (op_size_x - 1) + K_w - local_PL - local_PR;
            variant.registers_.workload_size0.workload_size0_bf.workload_size_y = stride_h * (op_size_y - 1) + K_h - local_PT - local_PB;
            variant.registers_.workload_size1.workload_size1_bf.workload_size_z = ConfigWorkloadSize(op_size_z);

            variant.registers_.weight_num = op_size_z;

            variant.registers_.workload_size1.workload_size1_bf.pad_count_up = local_PT;
            variant.registers_.workload_size1.workload_size1_bf.pad_count_down = local_PB;
            variant.registers_.workload_size1.workload_size1_bf.pad_count_left = local_PL;
            variant.registers_.workload_size1.workload_size1_bf.pad_count_right = local_PR;

            variant.registers_.workload_start0.workload_start0_bf.workload_start_x = (output_start_x * stride_w) - global_PL + local_PL;;
            variant.registers_.workload_start0.workload_start0_bf.workload_start_y = (output_start_y * stride_h) - global_PT + local_PT;
            variant.registers_.workload_start1.workload_start1_bf.workload_start_z = ConfigWorkloadStart(output_start_z);

            variant.registers_.te_beg1.te_beg1_bf.te_beg_x = output_start_x;
            variant.registers_.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
            variant.registers_.te_beg0.te_beg0_bf.te_beg_z = output_start_z;

            variant.registers_.te_end1.te_end1_bf.te_end_x = output_end_x;
            variant.registers_.te_end0.te_end0_bf.te_end_y = output_end_y;
            variant.registers_.te_end0.te_end0_bf.te_end_z = output_end_z;

            variant.weight_table_offset_ = output_start_z;

            //Split over K, and also streaming over K for now ....
            //ODU has a view of the full output tensor, yet as an optimization
            //in each cluster we bring weights and weight_table portions for each
            //output channel subset we compute in that particular cluster
            if (fb_invariant_->output_data()->dimensions()->Get(Z) != fb_invariant_->parent_output_tensor()->dimensions()->Get(Z))
            {
#ifdef NN_FATHOM_WORKAROUND_OUT_CHANNEL_OFFSET
                // Fathom style split logic. out_channel_offset may be set when it is not needed
                // Split calculation logic may not be correct, in flux from Fathom

                if (fb_invariant_->output_data()->locale_index()->size() > 1)
                {
                    // nnLog(MVLOG_INFO, "Fathom split weight table update: %u + %u (%u and %u)", variant.weight_table_offset_,
                    // fb_invariant_->out_channel_offset() * outChannelStride, fb_invariant_->out_channel_offset(), outChannelStride);
                    // variant.weight_table_offset_ += fb_invariant_->out_channel_offset() * outChannelStride;

                    nnLog(MVLOG_DEBUG,
                        "Using symmetric SoK: %u instead of %u",
                        variant.weight_table_offset_ % fb_invariant_->output_data()->dimensions()->Get(Z),
                        variant.weight_table_offset_);

                    variant.weight_table_offset_ %= fb_invariant_->output_data()->dimensions()->Get(Z);
                }
                else
                {
                    // Fathom can split an output among invariants but weights don't need to be adjusted
                    // The blob has the correct offsets already
                    // TODO: What about Fathom blobs which are really SOH/SOK. Is the above logic sufficient?
                    nnLog(MVLOG_WARN, "Invariant Z dim different than parent but no slices to broadcast");
                }
#else
                // MCM style split logic
                if (reinterpret_cast<const flatbuffers::Table *>(fb_invariant_)->CheckField(MVCNN::NCEInvariantFields::VT_OUT_CHANNEL_OFFSET))
                {
                    nnLog(MVLOG_DEBUG,
                        "Using asymmetric SoK: %u instead of %u",
                        variant.weight_table_offset_ - fb_invariant_->out_channel_offset(),
                        variant.weight_table_offset_ % fb_invariant_->output_data()->dimensions()->Get(Z));

                    variant.weight_table_offset_ -= fb_invariant_->out_channel_offset();
                }
                else
                {
                    nnLog(MVLOG_DEBUG,
                        "Using symmetric SoK: %u instead of %u",
                        variant.weight_table_offset_ % fb_invariant_->output_data()->dimensions()->Get(Z),
                        variant.weight_table_offset_);

                    variant.weight_table_offset_ %= fb_invariant_->output_data()->dimensions()->Get(Z);
                }
#endif
            }

            //Point into the 16 byte weight table entry, corresponding to the output channels subset
            variant.weight_table_offset_ = variant.weight_table_offset_ << 4;
            variant.output_sparsity_offset_ = invariant.output_sparsity_offset_  + fb_invariant_->odu_offset();
            float bpp = getODUDTypeSizeBits((nn::inference_runtime::OutputTensorDType) invariant.registers_.odu_cfg.odu_cfg_bf.dtype) / (float)CHAR_BIT;
            if(((unsigned int)(bpp * op_size_z) != (bpp * op_size_z)) || ((unsigned int)(bpp * op_size_z) % 16))
            {
                nnLog(MVLOG_ERROR, "Variant out Z is not multiple of 16bytes");
                return false;
            }

            op_size_y = SetupVariant_SOH(fb_invariant_, fb_variant_, invariant, variant, cluster_count_);
            invariant.output_sparsity_offset_ += (unsigned int)(op_size_y * op_size_x * op_size_z * bpp);

            bool is_out_dense = fb_invariant_->output_data()->data()->sparsity_index() == DEFAULT_INDEX;
            if(!is_out_dense)
                variant.output_sparsity_offset_ |= fb_invariant_->output_data()->locale_index()->Get(0) << 1;

//            nnLog(MVLOG_DEBUG, "Workload start/size, tensor beg/end: %u/%u, %u/%u",
//                   variant.registers_.workload_start0.workload_start0_bf.workload_start_y,
//                   variant.registers_.workload_size0.workload_size0_bf.workload_size_y,
//                   variant.registers_.te_beg0.te_beg0_bf.te_beg_y,
//                   variant.registers_.te_end0.te_end0_bf.te_end_y);

            variant.cluster_ = invariant.cluster_;

            variant.skip_dpu_computation_ = fb_variant_->mpe_mode() == MVCNN::MPE_Mode::MPE_Mode_NOP;

            return true;
        }

        bool DPUConfig::apply_workaround_for_bug_33243(DPUInvariantRegisters& registers)
        {
            if (workaround_for_bug_33243())
            {
                auto &cfg_invar = registers;

                // The following code is taken from https://github.com/movidius/mdk/pull/11773
                cfg_invar.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment = 1;

                uint32_t segment_size = SOH_LinesPerCluster(fb_invariant_->parent_input_tensor()->dimensions()->Get(Y), fb_invariant_->input_data()->dimensions()->Get(Y), cluster_count_);
                segment_size *= fb_invariant_->parent_input_tensor()->strides()->Get(STRIDES(Y));

                nnLog(MVLOG_DEBUG, "y: %u, x: %u, z: %u, seg_size: %u",
                        fb_invariant_->parent_input_tensor()->dimensions()->Get(Y),
                        fb_invariant_->parent_input_tensor()->dimensions()->Get(X),
                        fb_invariant_->parent_input_tensor()->dimensions()->Get(Z),
                        segment_size);

                // Copy from Zoran's code:
                // Fix for bug 33243 Computing results for cluster N we'll need the overlap
                // lines from cluster N+1 and read requests for elements near the left padding
                // in cluster N +1 will be invalid thus giving wrong results. Seq_size is
                // reduced (with tensor_z) to ensure reads for elements near left padding are
                // being issued to the right cluster. In order to not change buffer allocation
                // we must also modify se_addr* to account for the subtraction in se_seg_size
                uint32_t segment_offset = 0;

                // Issue is reproducing in hw and mitigated by compiler for DW IDU
                // with SOH enabled and odd left padding
                if ((fb_variant_->workload_start_X() == 0) && (fb_variant_->padLeft() % 2) &&
                // Fix should be applied in cases where the workload does not extend
                // to the rightmost input element
                    (fb_variant_->padRight() == 0) &&
                    ((uint32_t)(fb_variant_->workload_end_X() + 1) < fb_invariant_->parent_output_tensor()->dimensions()->Get(X))
                   )
                {
                    segment_offset = fb_invariant_->parent_input_tensor()->dimensions()->Get(Z);
                    nnLog(MVLOG_DEBUG, "Applying DW SOW workaround");
                }

                cfg_invar.se_sp_size[0].se_sp_size_bf.se_seg_size = (segment_size >> 4) - (segment_offset >> 4);
                cfg_invar.se_sp_size[1].se_sp_size_bf.se_seg_size = segment_size >> 4;
                cfg_invar.se_sp_size[2].se_sp_size_bf.se_seg_size = segment_size >> 4;

                // auto base_addr = act.data_addr;
                // uint32_t segment_addr = ((base_addr & 0x000FFFFF) + 0x3E000000);

                // Regardless of the cluster current wload is assigned, IDU will have a view
                // of the full input tensor and also a view into each cluster. That way he'll
                // stick to using mostly the buffer assigned to his direct cluster, expect for
                // the overlap cases when he'll use the neighbouring cluster buffers to get
                // the additional lines.

                // used only by cluster 0, except for overlap where cluster 1 can read the
                // extra line
                // This will be set later during the invariant update
                // cfg_invar.tensor_start = segment_addr;

                // Set here only the offsets, the base addresses will be added to them during the update
                // used only by cluster 1, except for overlap where cluster 0,2 can read the extra line
                cfg_invar.se_sp_addr[1].se_addr = - segment_offset;

                // used only by cluster 2, except for overlap where cluster 1,3 can read the extra line
                cfg_invar.se_sp_addr[2].se_addr = - segment_offset;

                // used only by cluster 3, except for overlap where cluster 2 can read the extra line
                cfg_invar.se_sp_addr[3].se_addr = - segment_offset;

                nnLog(MVLOG_DEBUG, "SOH Segmenting to: 0x%lx, 0x%lx, 0x%lx, 0x%lx",
                    (uint32_t)cfg_invar.tensor_start, (uint32_t)cfg_invar.se_sp_addr[1].se_addr,
                    (uint32_t)cfg_invar.se_sp_addr[2].se_addr, (uint32_t)cfg_invar.se_sp_addr[3].se_addr);
            }

            return true;
        }

        void DPUConfig::apply_16x1_grid_limit(dpu_runtime::DPUInvariantRegisters& registers, const char *layer)
        {
            if (registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign != dpu_runtime::MPE_GRID_16x1)
            {
                nnLog(MVLOG_WARN, "%s only supports MPE_GRID_16x1", layer);
                registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_16x1;
            }

            if (registers.odu_cfg.odu_cfg_bf.grid != dpu_runtime::ODU_GRID_16x1 &&
                registers.odu_cfg.odu_cfg_bf.grid != dpu_runtime::ODU_GRID_4x1)
            {
                nnLog(MVLOG_WARN, "%s only supports ODU_GRID_16x1 or ODU_GRID_4x1", layer);
                registers.odu_cfg.odu_cfg_bf.grid =
                    registers.odu_cfg.odu_cfg_bf.dtype == static_cast<unsigned int>(OutputTensorDType::FP16) ?
                    dpu_runtime::ODU_GRID_4x1 : dpu_runtime::ODU_GRID_16x1;
            }
        }

        bool DPUConfig::Setup_Input(dpu_runtime::DPUInvariantRegisters& registers)
        {
            const auto input = fb_invariant_->input_data();
            if (!input || !input->data() || !input->dimensions() || !input->strides())
            {
                nnLog(MVLOG_ERROR, "Missing input data");
                return false;
            }

            // Input Size
            registers.tensor_size0.tensor_size0_bf.tensor_size_x = input->dimensions()->Get(X);
            registers.tensor_size0.tensor_size0_bf.tensor_size_y = input->dimensions()->Get(Y);
            registers.tensor_size1.tensor_size1_bf.tensor_size_z = input->dimensions()->Get(Z);

            // SE pointers with 9 bit base index.
            registers.tensor_mode.tensor_mode_bf.addr_format_sel = 1;

            auto amode = input->data_dtype();
            auto dtype = ConfigDtype(amode);

            if (dtype == static_cast<uint8_t>(InputTensorDType::INPUT_DTYPE_UNKNOWN))
                return false;

            registers.tensor_mode.tensor_mode_bf.amode = dtype;

            if (amode == MVCNN::DType::DType_U8)
            {
                registers.mpe_cfg.mpe_cfg_bf.mpe_actbias =
                    (input->quant_zero() && input->quant_zero()->size()) ? input->quant_zero()->Get(0) : 0;

                registers.tensor_mode.tensor_mode_bf.pad_value =
                    (input->quant_zero() && input->quant_zero()->size()) ? input->quant_zero()->Get(0) : 0;
            }

            bool is_act_dense = input->data()->sparsity_index() == DEFAULT_INDEX;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = is_act_dense;

            SetupInvariant_Input(registers, input);

            return true;
        }

        bool DPUConfig::Setup_Weights(dpu_runtime::DPUInvariantRegisters& registers)
        {
            registers.weight_size = fb_invariant_->activation_window_channel_length();

            if (opType_ == DPU_MAXPOOL)
            {
                registers.tensor_mode.tensor_mode_bf.wmode = static_cast<unsigned int>(InputTensorDType::I8);
                return true;
            }

            const auto weights = fb_invariant_->weights_data();
            if (!weights || !weights->data() || !weights->dimensions() || !weights->strides())
            {
                nnLog(MVLOG_ERROR, "Missing weights data");
                return false;
            }

            auto wmode = weights->data_dtype();
            auto dtype = ConfigDtype(wmode);

            if (dtype == static_cast<uint8_t>(InputTensorDType::INPUT_DTYPE_UNKNOWN))
                return false;

            registers.tensor_mode.tensor_mode_bf.wmode = dtype;

            if (wmode == MVCNN::DType::DType_U8)
                registers.mpe_cfg.mpe_cfg_bf.mpe_wtbias =
                    (weights->quant_zero() && weights->quant_zero()->size()) ? weights->quant_zero()->Get(0) : 0;

            if (opType_ == DPU_ELTWISE)
                SetupInvariant_Input(registers, weights);

            return true;
        }

        bool DPUConfig::Setup_Kernel(dpu_runtime::DPUInvariantRegisters& registers)
        {
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign =
                (fb_invariant_->mpe_frequent_mode() == MVCNN::MPE_Mode_VECTOR) ? MPE_GRID_16x1 : MPE_GRID_4x4;

            if (fb_invariant_->kernelW() < KERNEL_SIZE_MIN || fb_invariant_->kernelW() > KERNEL_SIZE_MAX)
            {
                nnLog(MVLOG_ERROR, "Kernel width %u outside of supported range [%u-%u]",
                    fb_invariant_->kernelW(), KERNEL_SIZE_MIN, KERNEL_SIZE_MAX);

                return false;
            }

            registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x = fb_invariant_->kernelW();

            if (fb_invariant_->kernelH() < KERNEL_SIZE_MIN || fb_invariant_->kernelH() > KERNEL_SIZE_MAX)
            {
                nnLog(MVLOG_ERROR, "Kernel height %u outside of supported range [%u-%u]",
                    fb_invariant_->kernelH(), KERNEL_SIZE_MIN, KERNEL_SIZE_MAX);

                return false;
            }

            registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y = fb_invariant_->kernelH();

            if (fb_invariant_->kernel_strideW() != fb_invariant_->kernel_strideH())
                nnLog(MVLOG_WARN, "Only kernels with symmetric strides are supported. Received %ux%u, will use %u",
                    fb_invariant_->kernel_strideW(), fb_invariant_->kernel_strideH(), fb_invariant_->kernel_strideW());

            if (fb_invariant_->kernel_strideW() < KERNEL_STRIDE_MIN || fb_invariant_->kernel_strideW() > KERNEL_STRIDE_MAX)
            {
                nnLog(MVLOG_ERROR, "Kernel stride %u outside of supported range [%u-%u]",
                    fb_invariant_->kernel_strideW(), KERNEL_STRIDE_MIN, KERNEL_STRIDE_MAX);

                return false;
            }

            registers.tensor_mode.tensor_mode_bf.stride = fb_invariant_->kernel_strideW() - 1;

            registers.mpe_cfg.mpe_cfg_bf.mpe_mode =
                ConfigMpeActivationWeightDtype(registers.tensor_mode.tensor_mode_bf.amode, registers.tensor_mode.tensor_mode_bf.wmode);

            registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_left_en = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_right_en = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_top_en = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_bottom_en = 1;

            return true;
        }

        bool DPUConfig::Setup_Output(dpu_runtime::DPUInvariantRegisters& invariant)
        {
            const auto output = fb_invariant_->output_data();
            if (!output || !output->data() || !output->dimensions() || !output->strides())
            {
                nnLog(MVLOG_ERROR, "Missing output data");
                return false;
            }

            bool is_out_dense = output->data()->sparsity_index() == DEFAULT_INDEX;

            // This threshold size relies on using dense buffer dimensions.
            // TODO: update when starting to use shorter-than-dense buffers.
            invariant.odu_ctx_threshold = invariant.odu_ctx_size = 16777215; // max

            float bpp = (getODUDTypeSizeBits((nn::inference_runtime::OutputTensorDType)ConfigOutputDtype(output->data_dtype())) / (float)CHAR_BIT);
            auto se_size = output->data()->storage_element_size() ? output->data()->storage_element_size() : output->dimensions()->Get(Z);
            float se_size_in_bytes = se_size * bpp;
            if (!is_out_dense && (((unsigned int)se_size_in_bytes != se_size_in_bytes) || ((unsigned int)se_size_in_bytes % 16)))
            {
                nnLog(MVLOG_ERROR, "Workload storage element is not multiple of 16bytes");
                return false;
            }

            invariant.se_size = is_out_dense ? 0 : (((uint32_t)se_size_in_bytes) >> 4) - 1;

            auto dtype = ConfigOutputDtype(output->data_dtype());
            if (dtype == static_cast<uint8_t>(OutputTensorDType::OUTPUT_DTYPE_UNKNOWN))
                return false;

            invariant.odu_cfg.odu_cfg_bf.dtype = dtype;
            invariant.odu_cfg.odu_cfg_bf.ac_dense_mode = is_out_dense;// Activations are not/ packed per context

            invariant.odu_cfg.odu_cfg_bf.grid = (fb_invariant_->mpe_frequent_mode() == MVCNN::MPE_Mode_VECTOR) ? ODU_GRID_16x1 : ODU_GRID_4x4;

            if (invariant.odu_cfg.odu_cfg_bf.dtype == (unsigned int)OutputTensorDType::FP16)
                invariant.odu_cfg.odu_cfg_bf.grid = ODU_GRID_4x1;

            invariant.odu_cfg.odu_cfg_bf.adr_mode = 1;
            invariant.odu_cfg.odu_cfg_bf.write_ac = 1;//Always write data out!
            invariant.odu_cfg.odu_cfg_bf.write_pt = !is_out_dense;// Enable/Disable output SE table generation
            invariant.odu_cfg.odu_cfg_bf.write_sp = !is_out_dense;// Enable/Disable output sparsity map generation
            invariant.odu_cfg.odu_cfg_bf.sp_out_en = !is_out_dense;// Enable/Disable compression of output activations
            invariant.odu_cfg.odu_cfg_bf.order = output->strides()->Get(STRIDES(Z)) < output->strides()->Get(STRIDES(Y)) ? 1 : 0; // ZXY vs. XYZ
            invariant.odu_cfg.odu_cfg_bf.sp_value = output->quant_zero()->size() ? output->quant_zero()->Get(0) : 0;

            invariant.te_dim1.te_dim1_bf.te_dim_x = output->dimensions()->Get(X) - 1;
            invariant.te_dim0.te_dim0_bf.te_dim_y = output->dimensions()->Get(Y) - 1;

            // TODO: Why isn't this simply "output->dimensions()->Get(Z) - 1" ?
            // TODO: For channel-major output this seems to be incorrect
            float stride_x = output->strides()->Get(STRIDES(X));
            float stride_z = output->strides()->Get(STRIDES(Z));
            invariant.te_dim0.te_dim0_bf.te_dim_z = (stride_x / stride_z) - 1;

            SetupInvariant_Output(invariant, output);

#ifndef NN_FATHOM_WORKAROUND_ODU_OFFSET
            invariant.base_adr[0] = fb_invariant_->odu_offset();
#endif

            return true;
        }

        bool DPUConfig::SetupInvariant_RelativeAddresses(DPUAddresses& addresses)
        {
            if (const auto *input = fb_invariant_->input_data())
                if(!transform(*input, addresses.input_)) return false;

            if (const auto *output = fb_invariant_->output_data())
                if(!transform(*output, addresses.output_)) return false;

            if (const auto *wt = fb_invariant_->weights_table())
                if (!transform(*wt, addresses.weightsTable_)) return false;

            if (opType_ == nn::dpu_runtime::DPU_MAXPOOL)
                addresses.weights_ = addresses.input_;
            else if (const auto *wd = fb_invariant_->weights_data())
                if (!transform(*wd, addresses.weights_)) return false;

            if (const auto *ppe_task = fb_invariant_->ppe_task())
                if (const auto *il = ppe_task->instruction_list_data())
                    if (!transform(*il, addresses.ppe_list_)) return false;

            return true;
        }

        unsigned int DPUConfig::ConfigWorkloadSize(unsigned int size) const
        {
            switch (opType_)
            {
                case nn::dpu_runtime::DPU_CONV:
                case nn::dpu_runtime::DPU_ELTWISE:
                case nn::dpu_runtime::DPU_CMCONV:
                    // TODO: There seems to be some kind of untold convention with
                    // the compiler that this value will be overwritten in runtime
                    if (size != fb_invariant_->input_data()->dimensions()->Get(Z))
                        nnLog(MVLOG_DEBUG, "Op type %u does not support Z tiling. Got Zsize %u, using %u",
                            opType_, size, fb_invariant_->input_data()->dimensions()->Get(Z));

                    size = fb_invariant_->input_data()->dimensions()->Get(Z);
                    break;

                default:
                    break;
            }

            return size;
        }

        unsigned int DPUConfig::ConfigWorkloadStart(unsigned int start) const
        {
            switch (opType_)
            {
                case dpu_runtime::DPU_CONV:
                case dpu_runtime::DPU_ELTWISE:
                case dpu_runtime::DPU_CMCONV:
                    // TODO: There seems to be some kind of untold convention with
                    // the compiler that this value will be overwritten in runtime
                    if (start != 0)
                        nnLog(MVLOG_DEBUG, "Op type %u does not support Z tiling. Got Zstart %u, using 0",
                            opType_, start);

                    start = 0;
                    break;

                default:
                    break;
            }

            return start;
        }

        bool Update_Invariant(DPULayerTypes opType, DPUInvariant& invariant, const DPUAddresses &addresses, const NNRelocationData &relocationData)
        {
            auto input = addresses.input_;
            Update_Invariant_SOH(opType, invariant, input, relocationData);

            bool overflow = false;
            auto adrInput = input.resolve32(relocationData, RelativeAddress::Class::Data, &overflow);
            if (!adrInput)
            {
                nnLog(MVLOG_ERROR, "Workload input pointer is null");
                return false;
            }

            invariant.registers_.act_offset[0] = adrInput;

            if (!invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
            {
                invariant.registers_.se_sp_addr[0].se_addr = input.resolve32(relocationData, RelativeAddress::Class::SparsityTable, &overflow);
                invariant.registers_.se_sp_addr[0].sparsity_addr = input.resolve32(relocationData, RelativeAddress::Class::SparsityMap, &overflow);
            }

            auto adrOutput = addresses.output_.resolve32(relocationData, RelativeAddress::Class::Data, &overflow);
            if (!adrOutput)
            {
                nnLog(MVLOG_ERROR, "Workload output pointer is null");
                return false;
            }

            unsigned int offs[3] = { SLICE_LENGTH >> 4, SLICE_LENGTH >> 4, SLICE_LENGTH >> 4 };
            unsigned int base = RelativeAddress::to_dpu_multicast(adrOutput, offs[0], offs[1], offs[2]);

            if (base < invariant.registers_.base_adr[0])
            {
                // TODO: Is this a real error? Can it happen?
                nnLog(MVLOG_WARN, "Odu offset %u too large compared to base %u", invariant.registers_.base_adr[0], base);
            }

            invariant.registers_.base_adr[0] = base - invariant.registers_.base_adr[0];

            for (unsigned int i = 0; i < 3; ++i)
            {
                invariant.registers_.base_adr[i + 1] = invariant.registers_.base_adr[0];

                invariant.registers_.odu_cast[i].odu_cast_bf.cast_enable = offs[i] != 0;
                invariant.registers_.odu_cast[i].odu_cast_bf.cast_offset = offs[i];
            }

            if (invariant.registers_.odu_cfg.odu_cfg_bf.write_pt)
            {
                auto se_addr = addresses.output_.resolve32(relocationData, RelativeAddress::Class::SparsityTable, &overflow);
                invariant.registers_.pt_base = RelativeAddress::to_dpu_multicast_base(se_addr);
            }

            if (invariant.registers_.odu_cfg.odu_cfg_bf.write_sp)
            {
                auto sp_addr = addresses.output_.resolve32(relocationData, RelativeAddress::Class::SparsityMap, &overflow);
                invariant.registers_.sp_base = RelativeAddress::to_dpu_multicast_base(sp_addr);
            }

            switch (opType)
            {
                case DPU_CONV:
                case DPU_CMCONV:
                case DPU_MAXPOOL:
                case DPU_AVEPOOL:
                case DPU_DWCONV:
                {
                    RelativeAddress::Class weightClass =
#ifdef WORKAROUND_FORCE_WEIGHTS_OFFSET_TO_NNCMX_BASE_ADR
                        RelativeAddress::Class::Base;
#else
                        RelativeAddress::Class::Data;
#endif
                    auto adrWeights = addresses.weights_.resolve32(relocationData, weightClass, &overflow);
                    if ((adrWeights == 0) && ( opType != DPU_MAXPOOL))
                    {
                        nnLog(MVLOG_ERROR, "Workload weights pointer is null");
                        return false;
                    }

                    invariant.registers_.wt_offset = adrWeights;

                    auto adrWeightsTable = addresses.weightsTable_.resolve32(relocationData, RelativeAddress::Class::Data, &overflow);
                    if (!adrWeightsTable)
                    {
                        nnLog(MVLOG_ERROR, "Workload WeightsTable pointer is null");
                        return false;
                    }

                    invariant.registers_.weight_start = adrWeightsTable;

                    switch (opType)
                    {
                        case DPU_DWCONV:
                        case DPU_CMCONV:
                        case DPU_MAXPOOL:
                            invariant.registers_.tensor_start = adrInput;
                            break;

                        default:
                            break;
                    }

                    break;
                }

                case nn::dpu_runtime::DPU_ELTWISE:
                {
                    auto adrWeights = addresses.weights_.resolve32(relocationData, RelativeAddress::Class::Data, &overflow);
                    if (!adrWeights)
                    {
                        nnLog(MVLOG_ERROR, "Workload weights pointer is null");
                        return false;
                    }

                    if (!invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense)
                    {
                        invariant.registers_.elop_se_addr = addresses.weights_.resolve32(relocationData, RelativeAddress::Class::SparsityTable, &overflow);
                        invariant.registers_.elop_sparsity_addr = addresses.weights_.resolve32(relocationData, RelativeAddress::Class::SparsityMap, &overflow);

                        invariant.registers_.act_offset[0] = adrInput;
                        invariant.registers_.act_offset[1] = adrWeights;
                    }
                    else
                    {
                        //Dense Elops
                        //Start of Tensor A = adr0_offset[31:0] + [tensor_start[19:0],0000]
                        //Start of Tensor B = adr0_offset[31:0] + [weight_start[19:0],0000]
                        invariant.registers_.act_offset[0] = std::min(adrInput, adrWeights);
                        invariant.registers_.weight_start = (std::max(adrInput, adrWeights) - invariant.registers_.act_offset[0]) >> 4;
                    }

                    break;
                }

                default:
                    nnLog(MVLOG_ERROR, "Workload weights pointer is null");
                    return false;
            }

            if (addresses.ppe_list_.isValid())
            {
                if (auto adr = addresses.ppe_list_.resolve32(relocationData, RelativeAddress::Class::Data, &overflow).addr32())
                {
                    if (math::round_down<sizeof(unsigned int)>(adr) == math::round_up<sizeof(unsigned int)>(adr))
                        invariant.ppe_list_ = reinterpret_cast<const unsigned int *>(adr);
                    else
                        nnLog(MVLOG_WARN, "Unaligned PPE instruction list at %p. Ignoring", adr);
                }
            }

            invariant.allow_shadowing_ =
#ifndef NN_ENABLE_SPARSE_IDU_SHADOWING
                invariant.registers_.odu_cfg.odu_cfg_bf.ac_dense_mode &&
#endif
                // 1) https://hsdes.intel.com/appstore/article/#/18012316304 - FP16 workload shadowing halts for DW (and ZM) ops with odu multicast
                // 2) https://jira.devtools.intel.com/browse/EISW-14126 - FP16 workload shadowing halts for DW ops w/o odu multicast
                invariant.registers_.odu_cfg.odu_cfg_bf.dtype != static_cast<uint8_t>(OutputTensorDType::FP16);

            return !overflow;
        }
    }
}
