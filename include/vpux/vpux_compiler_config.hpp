//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @brief A header that defines advanced related properties for VPU compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 * Configuration API v1.0
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 * @file vpu_compiler_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"

#define VPU_COMPILER_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(NPU_COMPILER_##name)
#define VPU_COMPILER_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::NPU_COMPILER_##name

#define DECLARE_VPU_COMPILER_CONFIG_KEY(name) DECLARE_CONFIG_KEY(NPU_COMPILER_##name)
#define DECLARE_VPU_COMPILER_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(NPU_COMPILER_##name)

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for vpu compiler]
 * Describe log level for compiler
 * This option should be used with values: PluginConfigParams::LOG_INFO (default),
 * PluginConfigParams::LOG_ERROR, PluginConfigParams::LOG_WARNING,
 * PluginConfigParams::LOG_NONE, PluginConfigParams::LOG_DEBUG, PluginConfigParams::LOG_TRACE
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LOG_LEVEL);

/**
 * @brief [Only for vpu compiler and Level0 backend]
 * Type: "YES", "NO", default is "NO"
 * This option allows to perform FP32/FP16 to U8 input quantization on VPUX Plugin side via CPU
 * Note: current implementation will not allow us to use this feature in CiD case.
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(FORCE_HOST_QUANTIZATION);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
