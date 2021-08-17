/*
* {% copyright %}
*/
#pragma once

#include "sw_layer.h"
#include "layer_parser.h"

namespace nn {
namespace shave_lib {
class CustomLayerCppParser : public LayerParser {
  public:
    bool parse(const MVCNN::UPALayerTask *task, Layer *layer) override;

  private:
};
} // namespace shave_lib
} // namespace nn
