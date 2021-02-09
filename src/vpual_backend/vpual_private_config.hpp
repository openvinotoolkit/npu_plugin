//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <string>
#include <vpux/vpux_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

#define VPUX_VPUAL_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPUX_VPUAL_##name)
#define VPUX_VPUAL_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::VPUX_VPUAL_##name

#define DECLARE_VPUX_VPUAL_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPUX_VPUAL_##name)
#define DECLARE_VPUX_VPUAL_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPUX_VPUAL_##name)

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to force conversion of input from NCHW to NHWC ignoring TensorDesc info
 */
DECLARE_VPUX_VPUAL_CONFIG_KEY(REPACK_INPUT_LAYOUT);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use CoreNN plug-in for inference
 */
DECLARE_VPUX_VPUAL_CONFIG_KEY(USE_CORE_NN);

/**
 * @deprecated Use VPUX_VPUAL_USE_CORE_NN instead
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use CoreNN plug-in for inference
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_CORE_NN);

/**
 * @deprecated Use VPUX_INFERENCE_SHAVES instead
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 */
DECLARE_VPUX_VPUAL_CONFIG_KEY(INFERENCE_SHAVES);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
