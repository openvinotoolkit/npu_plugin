// {% copyright %}
#pragma once

#include "sw_layer.h"
#include "layer_parser.h"

namespace nn {
namespace shave_lib {
class PostOpsParser : public LayerParser {
  public:
    bool parse(const MVCNN::UPALayerTask *task, Layer *layer) override;
};
} // namespace shave_lib
} // namespace nn
