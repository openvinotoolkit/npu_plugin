//
// Copyright 2022 Intel Corporation.
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

#pragma once

#include <openvino/runtime/properties.hpp>

namespace ov {
namespace intel_vpux {

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/target"), default: "";
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> target_descriptor_path{"VPU_COMPILER_TARGET_DESCRIPTOR_PATH"};

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> target_descriptor{"VPU_COMPILER_TARGET_DESCRIPTOR"};

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("config/compilation"), default: "";
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> compilation_descriptor_path{"VPU_COMPILER_COMPILATION_DESCRIPTOR_PATH"};

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Default: "release_kmb";
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> compilation_descriptor{"VPU_COMPILER_COMPILATION_DESCRIPTOR"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable eltwise scales alignment
 * Configuration API v2.0
 */
static constexpr ov::Property<bool> eltwise_scales_alignment{"VPU_COMPILER_ELTWISE_SCALES_ALIGNMENT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable concat scales alignment
 * Configuration API v2.0
 */
static constexpr ov::Property<bool> concat_scales_alignment{"VPU_COMPILER_CONCAT_SCALES_ALIGNMENT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable weights zero points alignment
 * Configuration API v2.0
 */
static constexpr ov::Property<bool> weights_zero_points_alignment{"VPU_COMPILER_WEIGHTS_ZERO_POINTS_ALIGNMENT"};

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Provide path to custom layer binding xml file.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported.
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> custom_layers{"VPU_COMPILER_CUSTOM_LAYERS"};

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of comma separated group and pass values.
 * Removes {group, pass} value from mcm compilation descriptor.
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> compilation_pass_ban_list{"VPU_COMPILER_COMPILATION_PASS_BAN_LIST"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Enable or disable fusing scaleshift
 * Configuration API v2.0
 */
static constexpr ov::Property<bool> scale_fuse_input{"VPU_COMPILER_SCALE_FUSE_INPUT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 * Allow NCHW layout to be set as MCM Model input layout
 * Configuration API v2.0
 */
static constexpr ov::Property<bool> allow_nchw_mcm_input{"VPU_COMPILER_ALLOW_NCHW_MCM_INPUT"};

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 * Permute no-op layer can be used as dummy SW layer
 * Used as workaround in HETERO plugin
 * Configuration API v2.0
 */
static constexpr ov::Property<bool> remove_permute_noop{"VPU_COMPILER_REMOVE_PERMUTE_NOOP"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 * Configuration API v2.0
 */
static constexpr ov::Property<int64_t> inference_shaves{"VPUX_INFERENCE_SHAVES"};

/**
 * Type: Arbitrary string. Default is "-1".
 * This option allows to specify CSRAM size in bytes
 * When the size is -1, low-level SW is responsible for determining the required amount of CSRAM
 * When the size is 0, CSRAM isn't used
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> csram_size{"VPUX_CSRAM_SIZE"};

} // namespace intel_vpux
} // namespace ov
