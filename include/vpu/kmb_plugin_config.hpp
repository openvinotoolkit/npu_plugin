// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <string>

#include "ie_plugin_config.hpp"
#include "ie_api.h"

//
// Common options
//

#define VPU_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_##name)
#define VPU_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_##name

#define DECLARE_VPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_##name)
#define DECLARE_VPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_##name)

//
// KMB plugin options
//

#define VPU_KMB_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_KMB_##name)
#define VPU_KMB_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_KMB_##name

#define DECLARE_VPU_KMB_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_KMB_##name)
#define DECLARE_VPU_KMB_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_KMB_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

//
// KMB plugin options
//

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("ma2490"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("debug_ma2490"), default: "";
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB);
/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "NO".
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("."), default: "";
 * path where the mcmCompilator resulting files (blob, json, dot and png) should be placed
 * in folders named "<MCM_TARGET_DESCRIPTOR>/<MCM_COMPILATION_DESCRIPTOR>"
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH);

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string. Empty means ("<network name>"), default: "";
 * name of mcmCompilator resulting files (blob, json, dot and png)
 */
DECLARE_VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
