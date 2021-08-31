/*
* {% copyright %}
*/
#pragma once

#include <graphfile_generated.h>
#include <nn_inference_runtime_types.h>

namespace nn {
namespace shave_lib {

#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_MA2490_B0) || defined(CONFIG_TARGET_SOC_3100)
bool configSoftLayerTask(const MVCNN::UPALayerTask *task, nn::inference_runtime::frontend::SoftLayerTask *configTask);
#endif

} // namespace shave_lib
} // namespace nn
