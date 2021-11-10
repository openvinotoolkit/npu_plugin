/*
 * {% copyright %}
 */
#include <convert.h>

namespace parsing_lib {
void DPUConfigurator::SetupInvariant_MaxPool(DPUInvariantRegisters &registers) {
    registers.tensor_size0.tensor_size0_bf.tensor_size_x = srcInvariant.parent_input_tensor->dimensions[X];

    registers.tensor_mode.tensor_mode_bf.workload_operation = 2; // maxpool
    registers.tensor_mode.tensor_mode_bf.dw_input = 1;
    registers.tensor_mode.tensor_mode_bf.cm_input = 0;
    registers.tensor_mode.tensor_mode_bf.zm_input = 1;
    registers.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins = 1;
    registers.elops_wload.elops_wload_bf.pool_wt_rd_dis=1;

    SetupInvariant_SOH(registers);
}
} // namespace parsing_lib
