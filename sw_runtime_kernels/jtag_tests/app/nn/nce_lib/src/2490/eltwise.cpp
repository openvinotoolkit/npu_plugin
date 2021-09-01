/*
* {% copyright %}
*/
#include "nn_nce_lib.h"
#include "nn_nce_lib_conversion_fbs.h"
#include "nn_nce_lib_utils.h"

namespace nn
{
    namespace nce_lib
    {
        bool DPUConfig::SetupInvariant_Eltwise(dpu_runtime::DPUInvariantRegisters& registers)
        {
            if (fb_invariant_->kernelW() != 1 ||
                fb_invariant_->kernelH() != 1)
            {
                nnLog(MVLOG_ERROR, "Eltwise only supports 1x1 kernel. Got %ux%u", fb_invariant_->kernelW(), fb_invariant_->kernelH());
                return false;
            }

            if (fb_invariant_->parent_output_tensor() &&
                fb_invariant_->parent_output_tensor()->dimensions()->Get(Z) != fb_invariant_->output_data()->dimensions()->Get(Z))
            {
                nnLog(MVLOG_ERROR, "Eltwise does not support split over K\n");
                return false;
            }

            auto wt_tensor_ref = fb_invariant_->weights_data();

            auto amode = fb_invariant_->input_data()->data_dtype();
            auto wmode = fb_invariant_->weights_data()->data_dtype();
            auto omode = fb_invariant_->output_data()->data_dtype();
            registers.tensor_mode.tensor_mode_bf.zm_input = 1;//cm_input = 0,dw_input = 0;

            registers.odu_cfg.odu_cfg_bf.sp_in_en = 1;//element wise

            // For FP16 eltwise grid needs to be 4x4
            if(((amode == MVCNN::DType::DType_FP16) && (wmode == MVCNN::DType::DType_FP16)) || (omode == MVCNN::DType::DType_FP16))
            {
                registers.odu_cfg.odu_cfg_bf.grid = dpu_runtime::ODU_GRID_4x4;
                registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = dpu_runtime::MPE_GRID_4x4;
            }

            bool is_wt_dense = wt_tensor_ref->data()->sparsity_index() == DEFAULT_INDEX;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

            registers.weight_size =
                    wt_tensor_ref->dimensions()->Get(X) * wt_tensor_ref->dimensions()->Get(Y) * wt_tensor_ref->dimensions()->Get(Z);

            registers.elops_wload.elops_wload_bf.elop_wload = 1;//read in 2 tensors instead of a tensor and weight sets for a standard convolution.

            SetupInvariant_Input_SE_size(fb_invariant_, registers);

            return true;
        }
    }
}
