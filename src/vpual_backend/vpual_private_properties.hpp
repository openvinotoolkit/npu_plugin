//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>
#include <vpux/vpux_plugin_config.hpp>

namespace ov {
namespace intel_vpux {

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to force conversion of input from NCHW to NHWC ignoring TensorDesc info
 */
static constexpr Property<bool> repack_input_layout{"VPUX_VPUAL_REPACK_INPUT_LAYOUT"};

/**
 * @deprecated Use VPUX_INFERENCE_SHAVES instead
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 */
static constexpr Property<int64_t> vpual_inference_shaves{"VPUX_VPUAL_INFERENCE_SHAVES"};

}  // namespace intel_vpux
}  // namespace ov
