//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>

namespace ov {
namespace intel_vpux {

/**
 * @brief [Only for vpu compiler]
 * Type: std::string, default is empty.
 * Provide path to custom layer binding xml file.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported.
 * Configuration API v2.0
 */
static constexpr ov::Property<std::string> custom_layers{"VPU_COMPILER_CUSTOM_LAYERS"};

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

/**
 * @brief [Only for VPUX compiler]
 * Type: "YES", "NO", default is platform-dependent.
 * This option allows to use host based pre- and post- processing
 *
 * Note: Not only the preprocessing operations that are present in the
 * nGraph model are removed from IR, but also the ones introduced in the compiler itself.
 */
static constexpr ov::Property<bool> force_host_precision_layout_conversion{
        "VPUX_FORCE_HOST_PRECISION_LAYOUT_CONVERSION"};

}  // namespace intel_vpux
}  // namespace ov
