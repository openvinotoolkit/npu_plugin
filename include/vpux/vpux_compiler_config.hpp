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

#define VPU_COMPILER_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPU_COMPILER_##name)
#define VPU_COMPILER_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::VPU_COMPILER_##name

#define DECLARE_VPU_COMPILER_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_COMPILER_##name)
#define DECLARE_VPU_COMPILER_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_COMPILER_##name)

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR);

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
 * Enable or disable weights zero points alignment
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(WEIGHTS_ZERO_POINTS_ALIGNMENT);

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Provide path to custom layer binding xml file.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported.
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS);

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of comma separated group and pass values.
 * Removes {group, pass} value from mcm compilation descriptor.
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable fusing scaleshift
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(SCALE_FUSE_INPUT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 * Allow NCHW layout to be set as MCM Model input layout
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Permute no-op layer can be used as dummy SW layer
 * Used as workaround in HETERO plugin
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(REMOVE_PERMUTE_NOOP);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
