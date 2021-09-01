/*
* {% copyright %}
*/
#include <limits.h>
#include <nn_nce_lib_utils.h>

#include "nn_nce_lib_conversion_fbs.h"
#include <nn_runtime_types.h>

#if defined(CONFIG_TARGET_SOC_MA2490)
#if defined(REVISION_A0)
#define NN_WORKAROUND_33184_A0
#endif
#define NN_WORKAROUND_DENSE_SOH_A0
#endif

namespace nn
{
    namespace nce_lib
    {
        using namespace dpu_runtime;

        uint32_t calc_se_size(uint32_t x)
        {
            uint32_t sz = 0;
            uint32_t x_orig = x;
            while (x) {
                x >>= 1;
                ++sz;
            }
            if(x_orig != (1u << (sz - 1)))
            {
                nnLog(MVLOG_WARN, "storage_element_size is %d which is not a power of 2", x_orig);
            }
            // HW register NCE_DPU_Z_CONFIG.se_z_split has values: 1=16, 2=32....9=4096, 0=8192
            if (sz>4) {
                sz -= 4;
            } else {
                sz = 1;
            }
            // if Z size is 8192 or bigger, adjust to HW value for 8192
            if(sz >= 10)
            {
                sz = 0;
                nnLog(MVLOG_WARN, "storage_element_size bigger then 8192, HW value adjusted for 8192");
            }
            return sz;
        }

        void SetupInvariant_Input_SE_size(const MVCNN::NCEInvariantFields *wl_invariant, DPUInvariantRegisters &invariantRegisterst)
        {
            auto in_tensor = wl_invariant->input_data();
            bool is_act_dense = in_tensor->data()->sparsity_index() == DEFAULT_INDEX;
            auto in_dim_z = in_tensor->dimensions()->Get(Z);
            auto se_size = in_tensor->data()->storage_element_size();


            if(!is_act_dense && in_tensor->data()->storage_element_size())
            {
                if((wl_invariant->dpu_task_type() == MVCNN::DPULayerType_ELTWISE) &&  (se_size != in_dim_z))
                    nnLog(MVLOG_WARN, "storage_element_size set inside blob for eltwise != Z dim ---- not tested");
                // Z should be a power of 2
                invariantRegisterst.z_config.z_config_bf.se_z_split = calc_se_size(se_size);
                // num storage elements offset by 1 in HW. 0 = 1 SE Per Z direction
                invariantRegisterst.z_config.z_config_bf.num_ses_in_z_dir = (in_dim_z / se_size) - 1;
                if(in_dim_z % se_size)
                {
                    invariantRegisterst.z_config.z_config_bf.num_ses_in_z_dir++;
                    nnLog(MVLOG_WARN, "Z not divisible with SE size");
                }
            }
        }

        unsigned int SOH_LinesPerCluster(unsigned int parentHeight, unsigned int height, unsigned int clusters)
        {
            unsigned int lines_per_cluster = (parentHeight + clusters - 1) / clusters;

            if (height < lines_per_cluster)
                lines_per_cluster = (parentHeight - height) / (clusters - 1);
            else
                lines_per_cluster = height;

            return lines_per_cluster;
        }

        void SetupInvariant_SOH(const MVCNN::NCEInvariantFields *wl_invariant,
                                DPUInvariantRegisters &invariantRegisters, uint32_t clusters)
        {
            auto in_tensor = wl_invariant->input_data();
            auto in_parent = wl_invariant->parent_input_tensor();

            if (wl_invariant->is_segmented() && clusters > 1)
            {
                uint32_t seg_size = 0, sp_size = 0;
                bool is_act_dense = in_tensor->data()->sparsity_index() == DEFAULT_INDEX;
                unsigned int lines_per_cluster = SOH_LinesPerCluster(in_parent->dimensions()->Get(Y),
                    in_tensor->dimensions()->Get(Y), clusters);

                if (invariantRegisters.tensor_mode.tensor_mode_bf.dw_input)
                {
                    // maxpool & dw_conv
                    seg_size = lines_per_cluster * in_parent->strides()->Get(STRIDES(Y));
                }
                else
                {
                    // zm conv
                    seg_size = in_parent->dimensions()->Get(X) *
                               lines_per_cluster;
                    sp_size =  in_parent->dimensions()->Get(X) *
                               lines_per_cluster *
                               in_parent->dimensions()->Get(Z) >> 3;

                    const uint8_t dTypeSizeBits = getIDUDTypeSizeBits((nn::inference_runtime::InputTensorDType) invariantRegisters.tensor_mode.tensor_mode_bf.amode);
                    if (dTypeSizeBits < CHAR_BIT) {
                        const float bpp = dTypeSizeBits / (float)CHAR_BIT;
                        sp_size = sp_size * bpp;
                    }
                }

                for (uint32_t i = 0; (i < clusters - 1) && (i < 3); i++)
                {
                    if (invariantRegisters.tensor_mode.tensor_mode_bf.dw_input)
                    {
                        invariantRegisters.se_sp_size[i].se_sp_size_bf.se_seg_size = seg_size >> 4;
                    }
                    else
                    {
                        if (is_act_dense)
                        {
                            invariantRegisters.se_sp_size[i].se_sp_size_bf.se_seg_size = seg_size >> 2;
                        }
                        else
                        {
                            invariantRegisters.se_sp_size[i].se_sp_size_bf.se_seg_size = seg_size;
                            invariantRegisters.se_sp_size[i].se_sp_size_bf.sp_seg_size = sp_size >> 4;
                        }
                    }
                }

                invariantRegisters.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment = 1;
            }
        }

        unsigned int SetupVariant_SOH(const MVCNN::NCEInvariantFields *fb_invariant, const MVCNN::NCEVariantFields *fb_variant,
                                      DPUInvariant &invariant, DPUVariant &variant, unsigned int clusters)
        {
            auto output_start_y = fb_variant->workload_start_Y();
            auto output_end_y = fb_variant->workload_end_Y();

            if (fb_invariant->output_data()->dimensions()->Get(Y) != fb_invariant->parent_output_tensor()->dimensions()->Get(Y))
            {
                unsigned int lines_per_cluster = SOH_LinesPerCluster(fb_invariant->parent_output_tensor()->dimensions()->Get(Y), fb_invariant->output_data()->dimensions()->Get(Y), clusters);

                output_start_y %= lines_per_cluster;
                output_end_y %= lines_per_cluster;
                variant.registers_.te_beg0.te_beg0_bf.te_beg_y = output_start_y;
                variant.registers_.te_end0.te_end0_bf.te_end_y = output_end_y;

                if (((unsigned)output_start_y>fb_invariant->output_data()->dimensions()->Get(Y)) ||
                     ((unsigned)output_end_y>fb_invariant->output_data()->dimensions()->Get(Y)))
                    nnLog(MVLOG_WARN, "SOH workload still too big: %u-%u, tensor dim_y %lu", output_start_y, output_end_y, fb_invariant->output_data()->dimensions()->Get(Y));

                // Workload start needs adjustment if SOH was not set in invariant
                if (invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment == 0)
                {
                    auto stride_h = fb_invariant->kernel_strideH();
                    auto global_PT = fb_invariant->kernel_padTop();
                    auto local_PT = fb_variant->padTop();
                    variant.registers_.workload_start0.workload_start0_bf.workload_start_y = (output_start_y * stride_h) - global_PT + local_PT;
                }
            }

            return output_end_y - output_start_y + 1;
        }

        void SetupInvariant_Input(dpu_runtime::DPUInvariantRegisters &registers, const MVCNN::TensorReference *tensor)
        {
            auto base_ptrs = tensor->base_ptrs();
            if (base_ptrs->size() == MAX_CLUSTERS)
            {
                registers.base_offset_a |= (base_ptrs->Get(0) << 0) | (base_ptrs->Get(1) << 9);
                registers.base_offset_b |= (base_ptrs->Get(2) << 0) | (base_ptrs->Get(3) << 9);
            }
        }

        void SetupInvariant_Output(dpu_runtime::DPUInvariantRegisters &registers, const MVCNN::TensorReference *tensor)
        {
            auto base_ptrs = tensor->base_ptrs();
            if (base_ptrs->size() == MAX_CLUSTERS)
            {
                registers.base_ptr_a |= (base_ptrs->Get(0) << 9) | (base_ptrs->Get(1) << 0);
                registers.base_ptr_b |= (base_ptrs->Get(2) << 9) | (base_ptrs->Get(3) << 0);
            }
        }

        void Update_Invariant_SOH(dpu_runtime::DPULayerTypes opType, dpu_runtime::DPUInvariant &invariant, inference_runtime::RelativeAddress &input, const inference_runtime::NNRelocationData &relocationData)
        {
            if (invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment)
            {
                // Split over H segmenting
                input.set_index(1);

#ifndef NN_WORKAROUND_DENSE_SOH_A0
                auto adrInput = input.resolve32(relocationData);
#else
                (void)opType;
#endif

                for (int i = 0; i < 3; i++)
                {
                    if (invariant.registers_.se_sp_size[i].se_sp_size_bf.se_seg_size)
                    {
                        input.set_index(1 << (i + 1));

                        if (invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
                        {
#ifndef NN_WORKAROUND_DENSE_SOH_A0
                            if (opType == DPU_CONV)
                                invariant.registers_.se_sp_addr[i + 1].se_addr += (input.resolve32(relocationData)-adrInput) >> 4;
                            else
#endif
                                invariant.registers_.se_sp_addr[i + 1].se_addr += input.resolve32(relocationData);
                        }
                        else
                        {
                            // HW issue (A0): se_addr for segments 2+ need and offset from the real address of the segment.
                            if (!invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
                                invariant.registers_.se_sp_addr[i+1].se_addr = input.resolve32(relocationData, RelativeAddress::Class::SparsityTable).addr32() +
#ifdef NN_WORKAROUND_33184_A0
                                (i+1)*invariant.registers_.se_sp_size[i].se_sp_size_bf.se_seg_size*12;
#else
                                0;
#endif

                            if (!invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
                                invariant.registers_.se_sp_addr[i + 1].sparsity_addr = input.resolve32(relocationData, RelativeAddress::Class::SparsityMap).addr32();

                            //Previous layers have set the ODU base select to the cluster index
                            //Need to have matching logic at IDU side
                            invariant.registers_.act_offset[i + 1] = input.resolve32(relocationData);
                        }
                    }
                }
                input.set_index(1);
            }
        }


        bool Is_Dtype_Mix_Supported(MVCNN::DType inputType, MVCNN::DType weightsType)
        {
            auto is_type_supported = [] (MVCNN::DType type)
            {
                return ((type == MVCNN::DType::DType_FP16) || (type == MVCNN::DType::DType_U8)  ||
                        (type == MVCNN::DType::DType_I8)   || (type == MVCNN::DType::DType_I4)  ||
                        (type == MVCNN::DType::DType_I2)   || (type == MVCNN::DType::DType_I4X) || (type == MVCNN::DType::DType_BIN));
            };

            if (!is_type_supported(inputType))
            {
                nnLog(MVLOG_ERROR, "Input data type %s not supported", EnumNameDType(inputType));
                return false;
            }

            if (!is_type_supported(weightsType))
            {
                nnLog(MVLOG_ERROR, "Weights data type %s not supported", EnumNameDType(weightsType));
                return false;
            }

            if (inputType == weightsType)
                return true;

            // Matrix of supported input & weights data type combinations
            const bool supportedMixedDataTypes[][7] =
            {
            // weights type ---->
            // fp16  u8  i8  i4  i2  i4x bin
                {1,  0,  1,  1,  1,  0,  1}, // input type
                {0,  1,  0,  0,  0,  0,  1}, //    |
                {0,  0,  1,  1,  1,  0,  1}, //    |
                {0,  0,  1,  1,  1,  0,  1}, //    |
                {0,  0,  1,  1,  1,  0,  1}, //    v
                {0,  0,  0,  1,  1,  1,  1},
                {1,  1,  1,  1,  1,  0,  1}
            };

            // Compute input/weights data type indexes for the above table
            auto convert_type_to_idx = [] (MVCNN::DType type)
            {
                if (type == MVCNN::DType::DType_FP16)
                    return 0;
                if (type == MVCNN::DType::DType_U8)
                    return 1;
                return (type - MVCNN::DType::DType_I8 + 2);
            };

            auto inputTypeIndex = convert_type_to_idx(inputType);
            auto weightsTypeIndex = convert_type_to_idx(weightsType);

            if (supportedMixedDataTypes[inputTypeIndex][weightsTypeIndex])
            {
                return true;
            }

            nnLog(MVLOG_ERROR, "Input %s & Weights %s data type mix NOT supported", EnumNameDType(inputType), EnumNameDType(weightsType));
            return false;
        }
    }
}
