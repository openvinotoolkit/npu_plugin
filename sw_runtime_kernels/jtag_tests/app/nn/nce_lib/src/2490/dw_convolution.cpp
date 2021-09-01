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
            registers.tensor_mode.tensor_mode_bf.dw_input = 1;

            apply_16x1_grid_limit(registers, "DW Conv");

            SetupInvariant_SOH(fb_invariant_, registers, cluster_count_);
        }
    }
}
