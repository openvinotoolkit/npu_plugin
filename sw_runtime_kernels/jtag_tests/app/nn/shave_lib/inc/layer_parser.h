/*
* {% copyright %}
*/
#pragma once

#include "sw_layer.h"
#include <graphfile_generated.h>
#include <nn_memory.h>
#include <sw_tensor_ref.h>

namespace nn {
namespace shave_lib {

class LayerParser {
  public:
    virtual bool parse(const MVCNN::UPALayerTask *task, Layer *layer);
    virtual bool parse(const MVCNN::SNNLayerTask *task, Layer *layer);

    constexpr unsigned int getParamID(MVCNN::SoftwareLayerParams param) { return static_cast<unsigned int>(param); }
    static const char *getParamName(unsigned int paramID);

  protected:
    bool parseUPATensors(const MVCNN::UPALayerTask *task, Layer *layer);
    memory::cache_aligned_vector<TensorRef> &
    parseInputs(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>> *gfInputs,
                 Layer *layer, NDOrder baseLineOrder = FULL_ND_NHWC);
    memory::cache_aligned_vector<TensorRef> &
    parseOutputs(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>> *gfOutputs,
                 Layer *layer, NDOrder baseLineOrder = FULL_ND_NHWC);
  private:
    memory::cache_aligned_vector<TensorRef> &
    parseTensors(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>> *gfInputs,
                 memory::cache_aligned_vector<TensorRef> &vec, NDOrder baseLineOrder = FULL_ND_NHWC);
};
} // namespace shave_lib
} // namespace nn
