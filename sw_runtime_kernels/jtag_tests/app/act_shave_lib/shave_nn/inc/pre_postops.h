// {% copyright %}
#pragma once

#include "sw_layer.h"

namespace nn {
namespace shave_lib {

extern "C" {
preambleImpl prePostOpsHWC;
preambleImpl prePostOpsCHW;
preambleImpl prePostOpsHCW;
}

} // namespace shave_lib
} // namespace nn
