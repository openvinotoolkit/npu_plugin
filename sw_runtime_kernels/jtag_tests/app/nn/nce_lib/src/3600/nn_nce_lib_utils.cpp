/*
* {% copyright %}
*/
#include <nn_nce_lib_utils.h>

#include "nn_nce_lib_conversion_fbs.h"
#include <nn_runtime_types.h>

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

            if ((in_parent->dimensions()->Get(Y) != in_tensor->dimensions()->Get(Y))
                && clusters > 1
            )
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

                        switch (i+1)
                        {
                            case 1: invariantRegisters.base_offset_a |= 0x1 << 9; break;
                            case 2: invariantRegisters.base_offset_b |= 0x2 << 0; break;
                            case 3: invariantRegisters.base_offset_b |= 0x3 << 9; break;
                        }
                    }
                }

                invariantRegisters.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment = 1;
            }
        }


        void SetupInvariant_SOH_Input(const MVCNN::NCEInvariantFields *wl_invariant,
                                              DPUInvariantRegisters &invariantRegisters)
        {
            auto in_tensor = wl_invariant->input_data();
            auto in_parent = wl_invariant->parent_input_tensor();

            bool is_act_dense = in_tensor->data()->sparsity_index() == DEFAULT_INDEX;

            if(!is_act_dense)
            {
                if ((invariantRegisters.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment == 0) &&
                    (in_parent->dimensions()->Get(Y) != in_tensor->dimensions()->Get(Y)))
                {
                    invariantRegisters.base_offset_a |= in_tensor->locale_index()->Get(0);
                }
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

                bool is_out_dense = fb_invariant->output_data()->data()->sparsity_index() == DEFAULT_INDEX;
                if(!is_out_dense)
                    variant.output_sparsity_offset_ |= fb_invariant->output_data()->locale_index()->Get(0) << 1;
            }

            return output_end_y - output_start_y + 1;
        }

        void Setup_Output_SOH(const MVCNN::NCEInvariantFields *fb_invariant, DPUInvariantRegisters &invariant, bool is_out_dense)
        {
            // TODO: remove these - copied directly from POC
            invariant.base_ptr_a = 0x1;
            invariant.base_ptr_b = 0x403;
            if (!is_out_dense && fb_invariant->output_data()->dimensions()->Get(Y) != fb_invariant->parent_output_tensor()->dimensions()->Get(Y))
            {
                invariant.base_ptr_a = (0 << 9) | (1 << 0);
                invariant.base_ptr_b = (2 << 9) | (3 << 0);
            }
        }

        void Update_Invariant_SOH(dpu_runtime::DPULayerTypes opType, dpu_runtime::DPUInvariant &invariant, inference_runtime::RelativeAddress &input, const inference_runtime::NNRelocationData &relocationData)
        {
            UNUSED(opType);
            if (invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment)
            {
                // Split over H segmenting
                input.set_index(1);

                for (int i = 0; i < 3; i++)
                {
                    if (invariant.registers_.se_sp_size[i].se_sp_size_bf.se_seg_size)
                    {
                        input.set_index(1 << (i + 1));

                        if (invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
                        {

                            invariant.registers_.se_sp_addr[i + 1].se_addr += input.resolve32(relocationData);
                        }
                        else
                        {
                            // HW issue (A0): se_addr for segments 2+ need and offset from the real address of the segment.
                            if (!invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
                                invariant.registers_.se_sp_addr[i+1].se_addr = input.resolve32(relocationData, RelativeAddress::Class::SparsityTable);

                            if (!invariant.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
                                invariant.registers_.se_sp_addr[i + 1].sparsity_addr = input.resolve32(relocationData, RelativeAddress::Class::SparsityMap);

                            //Previous layers have set the ODU base select to the cluster index
                            //Need to have matching logic at IDU side
                            invariant.registers_.act_offset[i + 1] = input.resolve32(relocationData);
                        }
                    }
                }
                input.set_index(1);
            }
        }

        void SetupVariant_NTHW_NTK(const MVCNN::NCEInvariantFields *fb_invariant, DPUVariantRegisters &variant)
        {
            auto mpe_frequent_mode = fb_invariant->mpe_frequent_mode();

            // Sets up on NTHW on IDU
            nnLog(MVLOG_DEBUG, "mpe_frequent_mode %u", mpe_frequent_mode);
            switch (mpe_frequent_mode)
            {
            case MVCNN::MPE_Mode_VECTOR:
                variant.offset_addr.offset_addr_bf.nthw_ntk = dpu_runtime::IDU_NTHW_NTK_8_8;
                break;
            case MVCNN::MPE_Mode_CUBOID_4x16: // NTH = 1, NTW=4, NTK = 16 (4, 16)
                variant.offset_addr.offset_addr_bf.nthw_ntk = dpu_runtime::IDU_NTHW_NTK_4_16;
                break;
            case MVCNN::MPE_Mode_CUBOID_8x16: // NTH = 2, NTW=4, NTK = 8 (8, 8)
                variant.offset_addr.offset_addr_bf.nthw_ntk = dpu_runtime::IDU_NTHW_NTK_8_8;
                break;
            case MVCNN::MPE_Mode_CUBOID_16x16: // NTH = 4, NTW=4, NTK = 4  (16, 4)
                variant.offset_addr.offset_addr_bf.nthw_ntk = dpu_runtime::IDU_NTHW_NTK_16_4;
                break;
            default:
                nnLog(MVLOG_ERROR, "mpe_frequent_mode %u", mpe_frequent_mode);
                assert(!"Non Supported Grid Type");
                break;
            }
        }

        void SetupInvariant_Grid(const MVCNN::NCEInvariantFields *fb_invariant, DPUInvariantRegisters &invariant)
        {
            auto mpe_frequent_mode = fb_invariant->mpe_frequent_mode();

            // Sets up on NTHW on IDU
            nnLog(MVLOG_DEBUG, "mpe_frequent_mode %u", mpe_frequent_mode);
            switch (mpe_frequent_mode)
            {
            case MVCNN::MPE_Mode_VECTOR:
                invariant.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_16x1;
                invariant.odu_cfg.odu_cfg_bf.nthw = dpu_runtime::ODU_NTHW_1;
                invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_16x1;;
                break;
            case MVCNN::MPE_Mode_CUBOID_4x16: // NTH = 1, NTW=4, NTK = 16 (4, 16)
                invariant.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_4x4;
                invariant.odu_cfg.odu_cfg_bf.nthw = dpu_runtime::ODU_NTHW_4;
                invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_4x4;
                break;
            case MVCNN::MPE_Mode_CUBOID_8x16: // NTH = 2, NTW=4, NTK = 8 (8, 8)
                invariant.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_4x4;
                invariant.odu_cfg.odu_cfg_bf.nthw = dpu_runtime::ODU_NTHW_8;
                invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_4x4;
                break;
            case MVCNN::MPE_Mode_CUBOID_16x16: // NTH = 4, NTW=4, NTK = 4  (16, 4)
                invariant.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_4x4;
                invariant.odu_cfg.odu_cfg_bf.nthw = dpu_runtime::ODU_NTHW_16;
                invariant.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_4x4;
                break;
            default:
                nnLog(MVLOG_ERROR, "mpe_frequent_mode %u", mpe_frequent_mode);
                assert(!"Non Supported Grid Type");
                break;
            }
        }
    }
}
