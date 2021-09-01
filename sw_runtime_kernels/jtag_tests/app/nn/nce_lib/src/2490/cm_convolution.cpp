/*
* {% copyright %}
*/
#include "nn_nce_lib.h"
#include "nn_nce_lib_conversion_fbs.h"

namespace nn
{
    namespace nce_lib
    {
        bool DPUConfig::SetupInvariant_CMConv(dpu_runtime::DPUInvariantRegisters& registers)
        {
            if (static_cast<unsigned int>(fb_invariant_->input_data()->strides()->Get(STRIDES(Y))) % 16)
                nnLog(MVLOG_WARN, "CM Conv requires Y stride to be multiple to 16. Received: %u",
                    fb_invariant_->input_data()->strides()->Get(STRIDES(Y)));

            // CM Conv requires act_dense == 0, even if activations are dense.
            registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = 0;

            registers.tensor_mode.tensor_mode_bf.cm_input = 1;

            apply_16x1_grid_limit(registers, "CM Conv");

            return true;
        }
    }
}
