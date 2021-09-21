/*
* {% copyright %}
*/
#include "layer_parser.h"
#include "sw_layer.h"
#include "sw_nn_runtime_types.h"
#include "tensor_gf_util.h"
//#include <nn_nce_lib_conversion_fbs.h>

#include <nn_log.h>
#include <nn_cache.h>

namespace nn {
namespace shave_lib {

bool LayerParser::parse(const MVCNN::UPALayerTask *task, Layer *layer) {
    (void)task;  // -Werror=unused-parameter
    (void)layer; // -Werror=unused-parameter
    nnLog(MVLOG_ERROR, "Layer %s is not implemented as a UPA layer",
         EnumNameSoftwareLayerParams(task->softLayerParams_type()));
    return false;
}

bool LayerParser::parse(const MVCNN::SNNLayerTask *task, Layer *layer) {
    (void)task;  // -Werror=unused-parameter
    (void)layer; // -Werror=unused-parameter
    nnLog(MVLOG_ERROR, "Layer %s is not implemented as a SNN layer",
          EnumNameSoftwareLayerParams(task->softLayerParams_type()));
    return false;
}

const char *LayerParser::getParamName(unsigned int paramID) { return MVCNN::EnumNamesSoftwareLayerParams()[paramID]; }

bool LayerParser::parseUPATensors(const MVCNN::UPALayerTask *task, Layer *layer) {
    if (task->inputs() && task->outputs()) {
        if (task->inputs()->size() > MAX_INPUT_TENSORS) {
            nnLog(MVLOG_ERROR, "Too many input tensors in graph!");
            return false;
        }
        if (task->outputs()->size() > MAX_OUTPUT_TENSORS) {
            nnLog(MVLOG_ERROR, "Too many output tensors in graph!");
            return false;
        }

        parseInputs(task->inputs(), layer);
        parseOutputs(task->outputs(), layer);
        return ((layer->getInputs()).size() > 0 && (layer->getOutputs()).size() > 0);
    } else {
        bool success = true;
        layer->getInputs().resize(3);
        layer->getOutputs().resize(1);

        auto itr = layer->getInputs();
        auto otr = layer->getOutputs();

        if (task->input_data()) {
            success = success && parseTensorRef(task->input_data(), &itr[0]);
        }
        if (success && task->weights_data()) {
            success = success && parseTensorRef(task->weights_data(), &itr[1]);
        }
        if (success && task->weights_table()) {
            success = success && parseTensorRef(task->weights_table(), &itr[2]);
        }
        if (success && task->output_data()) {
            success = success && parseTensorRef(task->output_data(), &otr[0]);
        }
        if (!success) {
            layer->getInputs().resize(0);
            layer->getOutputs().resize(0);
        }
        return success;
    }
}

memory::cache_aligned_vector<TensorRef> &
LayerParser::parseInputs(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>> *gfTensors,
                           Layer *layer, NDOrder baseLineOrder) {
    memory::cache_aligned_vector<TensorRef> &vecND = layer->getInputs();
    parseTensors(gfTensors, vecND, baseLineOrder);
    if (gfTensors->size() > 0 && vecND.size() == 0) {
        nnLog(MVLOG_ERROR, "Input tensors parsing fails");
    }

    return vecND;
}

memory::cache_aligned_vector<TensorRef> &
LayerParser::parseOutputs(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>> *gfTensors,
                           Layer *layer, NDOrder baseLineOrder) {
    memory::cache_aligned_vector<TensorRef> &vecND = layer->getOutputs();
    if (vecND.size() == 0) {
        parseTensors(gfTensors, vecND, baseLineOrder);
    }
    if (gfTensors->size() > 0 && vecND.size() == 0) {
        nnLog(MVLOG_ERROR, "Output tensors parsing fails");
    }

    return vecND;
}

memory::cache_aligned_vector<TensorRef> &
LayerParser::parseTensors(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>> *gfTensors,
                          memory::cache_aligned_vector<TensorRef> &vec, NDOrder baseLineOrder) {
    nnLog(MVLOG_DEBUG, "Parsing %d tensors\n", gfTensors->size());
    vec.resize(gfTensors->size());
    for (size_t i = 0; i < gfTensors->size(); i++) {
        if (!parseTensorRef(gfTensors->Get(i), &vec[i], baseLineOrder)) {
            vec.resize(0);
        }
    }

    return vec;
}

} // namespace shave_lib
} // namespace nn
