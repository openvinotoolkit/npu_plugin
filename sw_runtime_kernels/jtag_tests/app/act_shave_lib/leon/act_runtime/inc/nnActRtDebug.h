/*
 * {% copyright %}
 */
#pragma once

#include <nn_runtime_types.h>
#include "nn_fifo_manager.h"

using namespace nn::common_runtime;
using namespace nn::common_runtime::fifo;

namespace nn {
namespace act_runtime {

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
void execDebug(ActKernelRange *wl, unsigned int shaveIndex, SHVFifoConfig cfg);
#endif

} // namespace act_runtime
} // namespace nn
