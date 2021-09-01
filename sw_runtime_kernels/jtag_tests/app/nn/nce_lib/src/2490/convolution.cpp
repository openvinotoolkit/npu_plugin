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
        void DPUConfig::SetupInvariant_Convolution(dpu_runtime::DPUInvariantRegisters& registers)
        {
            auto in_tensor_ref = fb_invariant_->input_data();
            auto wt_tensor_ref = fb_invariant_->weights_data();

            registers.tensor_mode.tensor_mode_bf.zm_input = 1;

            bool is_wt_dense = wt_tensor_ref->data()->sparsity_index() == DEFAULT_INDEX;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

            registers.weight_size = in_tensor_ref->dimensions()->Get(Z) * fb_invariant_->kernelW() * fb_invariant_->kernelH();

            SetupInvariant_SOH(fb_invariant_, registers, cluster_count_);

            // Input Size
            if (registers.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment)
            {
                registers.tensor_size0.tensor_size0_bf.tensor_size_y = fb_invariant_->parent_input_tensor()->dimensions()->Get(Y);
                registers.tensor_size1.tensor_size1_bf.tensor_size_z = fb_invariant_->parent_input_tensor()->dimensions()->Get(Z);
                registers.tensor_size0.tensor_size0_bf.tensor_size_x = fb_invariant_->parent_input_tensor()->dimensions()->Get(X);
            }

            SetupInvariant_Input_SE_size(fb_invariant_, registers);
        }
    }
}
