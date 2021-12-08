/*
 * {% copyright %}
 */
#ifndef NN_NCE_LIB_IR_H_
#define NN_NCE_LIB_IR_H_

#include <nn_runtime_types.h>
#include <nn_relocation.h>
#include <nn_inference_runtime_types.h>

namespace nn {
namespace nce_lib {
#ifdef NN_PRINT_DPU_REGISTERS
void DebugPrintRegister(const dpu_runtime::DPUVariant &variant);
#endif
void dump_output(const uint8_t invar_idx, const dpu_runtime::DPUInvariant &invariant);
} // namespace nce_lib
} // namespace nn
#endif /* NN_NCE_LIB_IR_H_ */
