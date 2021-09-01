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
        void DPUConfig::SetupInvariant_DwConvolution(dpu_runtime::DPUInvariantRegisters& registers)
        {
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins = 1; // enable the IDU to generate the weight sparsity
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

            // Dedicated HW for channel-major removed for 2.7 - all convolutions
            // processed as Z-major. Depthwise here is processed as a subset of
            // the Z-major convolution.
            registers.tensor_mode.tensor_mode_bf.dw_input = 1;
            registers.tensor_mode.tensor_mode_bf.cm_input = 0;
            registers.tensor_mode.tensor_mode_bf.zm_input = 1;

            auto wt_tensor_ref = fb_invariant_->weights_data();
            bool is_wt_dense = wt_tensor_ref->data()->sparsity_index() == DEFAULT_INDEX;
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

            SetupInvariant_SOH(fb_invariant_, registers, cluster_count_);
        }
    }
}
