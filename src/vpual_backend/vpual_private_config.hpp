//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

/**
 *
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 */

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
 * Configuration API v1.0
 */
DECLARE_VPUX_VPUAL_CONFIG_KEY(REPACK_INPUT_LAYOUT);

/**
 * @deprecated Use VPUX_INFERENCE_SHAVES instead
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 * Configuration API v1.0
 */
DECLARE_VPUX_VPUAL_CONFIG_KEY(INFERENCE_SHAVES);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
