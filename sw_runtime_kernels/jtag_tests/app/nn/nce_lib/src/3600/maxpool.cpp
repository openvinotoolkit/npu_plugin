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
        void DPUConfig::SetupInvariant_MaxPool(dpu_runtime::DPUInvariantRegisters& registers)
        {
            registers.tensor_size0.tensor_size0_bf.tensor_size_x = fb_invariant_->parent_input_tensor()->dimensions()->Get(X);

            registers.tensor_mode.tensor_mode_bf.workload_operation = 2;//maxpool
            registers.tensor_mode.tensor_mode_bf.dw_input = 1;
            registers.tensor_mode.tensor_mode_bf.cm_input = 0;
            registers.tensor_mode.tensor_mode_bf.zm_input = 1;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins = 1;

            SetupInvariant_SOH(fb_invariant_, registers, cluster_count_);
        }
    }
}
