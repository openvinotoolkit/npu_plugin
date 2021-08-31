/*
* {% copyright %}
*/
#pragma once

#include "sw_layer.h"

namespace nn {
namespace shave_lib {

extern "C" {
preambleImpl preCustomLayerCpp;
execCleanupImpl  execCleanupCustomLayerCpp;
}

} // namespace shave_lib
} // namespace nn
