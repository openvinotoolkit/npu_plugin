/*
 * {% copyright %}
 */
#include "convert.h"

namespace parsing_lib {
void DPUConfigurator::SetupInvariant_Convolution(DPUInvariantRegisters &registers) {
    auto wt_tensor_ref = srcInvariant.weights_data;

    registers.tensor_mode.tensor_mode_bf.zm_input = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;

    bool is_wt_dense = wt_tensor_ref->data.sparsity_index == DEFAULT_INDEX;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = is_wt_dense;

    SetupInvariant_SOH(registers);
    SetupInvariant_SOH_Input(registers);

    // Input Size
    if (registers.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment) {
        registers.tensor_size0.tensor_size0_bf.tensor_size_y =
            srcInvariant.parent_input_tensor->dimensions[Y];
        registers.tensor_size1.tensor_size1_bf.tensor_size_z =
            srcInvariant.parent_input_tensor->dimensions[Z];
        registers.tensor_size0.tensor_size0_bf.tensor_size_x =
            srcInvariant.parent_input_tensor->dimensions[X];
    }

    SetupInvariant_Input_SE_size(registers);
}
}
