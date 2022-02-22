//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

#define VPU_COMPILER_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPU_COMPILER_##name)
#define VPU_COMPILER_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::VPU_COMPILER_##name

#define DECLARE_VPU_COMPILER_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_COMPILER_##name)
#define DECLARE_VPU_COMPILER_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_COMPILER_##name)

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR);

/**
 * @brief [Only for vpu compiler]
 * Describe log level for mcmCompiler
 * This option should be used with values: PluginConfigParams::LOG_INFO (default),
 * PluginConfigParams::LOG_ERROR, PluginConfigParams::LOG_WARNING,
 * PluginConfigParams::LOG_NONE, PluginConfigParams::LOG_DEBUG, PluginConfigParams::LOG_TRACE
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(LOG_LEVEL);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable eltwise scales alignment
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable concat scales alignment
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable weights zero points alignment
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(WEIGHTS_ZERO_POINTS_ALIGNMENT);

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Provide path to custom layer binding xml file.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported.
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS);

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of comma separated group and pass values.
 * Removes {group, pass} value from mcm compilation descriptor.
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable fusing scaleshift
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(SCALE_FUSE_INPUT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 * Allow NCHW layout to be set as MCM Model input layout
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Permute no-op layer can be used as dummy SW layer
 * Used as workaround in HETERO plugin
 * Configuration API v1.0
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(REMOVE_PERMUTE_NOOP);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
