/*
* {% copyright %}
*/
#include "sw_layer.h"
#include "nn_shave_lib.h"
#include "layer_loader.h"
#include "sw_shave_dispatcher.h"
#include <assert.h>
#include <nn_log.h>
#include <nn_cache.h>

namespace nn {
namespace shave_lib {

bool configSoftLayerTask(const MVCNN::UPALayerTask *task, nn::inference_runtime::frontend::SoftLayerTask *configTask)
{
    auto *layer = &configTask->layer_;
    if (reinterpret_cast<uintptr_t>(layer) % 64 != 0)
    {
        nnLog(MVLOG_ERROR, "Layer reference %p should be cache aligned", layer);
        return false;
    }

    new (layer) Layer; // It's not guaranteed that this memblock was constructed

    {
        auto success = LayerLoader::parseUPALayer(task, layer);

        if (!success)
        {
            nnLog(MVLOG_ERROR, "UPA SW layer parse failure %s", MVCNN::EnumNameSoftwareLayerParams(task->softLayerParams_type()));
            return false;
        }
    }
    {
        auto inputs = layer->params.inputs.size();
        auto outputs = layer->params.outputs.size();

        if (inputs >= MAX_INPUT_TENSORS)
        {
            nnLog(MVLOG_ERROR, "Too many input tensors: %u Only %u supported", inputs, MAX_INPUT_TENSORS);
            return false;
        }

        if (outputs >= MAX_OUTPUT_TENSORS)
        {
            nnLog(MVLOG_ERROR, "Too many output tensors: %u Only %u supported", outputs, MAX_OUTPUT_TENSORS);
            return false;
        }

        configTask->num_inputs_ = inputs;
        configTask->num_outputs_ = outputs;

        for (size_t i = 0; i < inputs; i++)
            configTask->rel_inputs_[i] = layer->params.inputs[i].dataAddr;

        for (size_t o = 0; o < outputs; o++)
            configTask->rel_outputs_[o] = layer->params.outputs[o].dataAddr;
    }

    configTask->is_trailing_layer = task->isTrailingSWLayer();

    nn::cache::flush(*configTask);

    return true;
}

} // namespace shave_lib
} // namespace nn
