/*
* {% copyright %}
*/
#pragma once

#include "sw_layer.h"
#include "sw_layer_params.h"
#include "sw_shave_lib_common.h"
#include "sw_shave_res_manager.h"
#include "sw_tensor_ref.h"

#include <nn_memory.h>
#include <nn_runtime_types.h>
#include <nn_inference_runtime_types.h>

namespace nn {
namespace shave_lib {

/**
 * This class is intended to abstract a set of SHAVE resources. For example, a
 * subclass of @LayerRunner for the UPA block is expected to:
 * 1) reserve/negotiate requested shave resources
 * 2) handle a global context of shaves allocated for inference between
 *    different inference threads
 * 3) launch a kernel using LNN and UPA Shave(s)
 * 4) block until execution is complete
 *
 * Opimization such as forgoing re-loading of a shave kernel and just blocking
 * if the same kernel is already loaded and executing for a different inf.
 * thread would be done here.
 */
class LayerRunner {
  public:
    virtual ~LayerRunner() = 0;
    /**
     * blocks until done
     */
    virtual void run(const Layer *layer, const AbsoluteAddresses *aba, unsigned int workerTID, unsigned int infTID, ShavePerfCounters *counters) = 0;
};

} // namespace shave_lib
} // namespace nn
