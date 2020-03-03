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


/**
 * @brief A header that defines advanced related properties for VPU compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_compiler_config.hpp
 */

#pragma once

#include <vpu/vpu_plugin_config.hpp>

#define VPU_COMPILER_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_COMPILER_##name)
#define VPU_COMPILER_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_COMPILER_##name

#define DECLARE_VPU_COMPILER_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_COMPILER_##name)
#define DECLARE_VPU_COMPILER_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_COMPILER_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("ma2490"), default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("debug_ma2490"), default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB);
/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(PARSING_ONLY);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_JSON);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_DOT);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("."), default: "";
 * path where the mcmCompilator resulting files (blob, json, dot and png) should be placed
 * in folders named "<TARGET_DESCRIPTOR>/<COMPILATION_DESCRIPTOR>"
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("<network name>"), default: "";
 * name of mcmCompilator resulting files (blob, json, dot and png)
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS);

/**
 * @brief [Only for vpu compiler]
 * Describe log level for mcmCompiler
 * This option should be used with values: PluginConfigParams::LOG_INFO (default),
 * PluginConfigParams::LOG_ERROR, PluginConfigParams::LOG_WARNING,
 * PluginConfigParams::LOG_NONE, PluginConfigParams::LOG_DEBUG, PluginConfigParams::LOG_TRACE
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LOG_LEVEL);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable eltwise scales alignment
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable concat scales alignment
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable Input->ScaleShift pattern removing
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(INPUT_SCALE_SHIFT_REMOVING);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable weights zero points alignment
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(WEIGHTS_ZERO_POINTS_ALIGNMENT);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
