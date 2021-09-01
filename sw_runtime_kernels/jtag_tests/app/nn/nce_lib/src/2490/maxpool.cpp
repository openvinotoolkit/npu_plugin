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
            registers.tensor_mode.tensor_mode_bf.workload_operation = 2;//maxpool
            registers.tensor_mode.tensor_mode_bf.dw_input = 1;

            apply_16x1_grid_limit(registers, "Maxpool");

            SetupInvariant_SOH(fb_invariant_, registers, cluster_count_);
        }
    }
}
