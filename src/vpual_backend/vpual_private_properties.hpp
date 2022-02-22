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

#include <string>
#include <vpux/vpux_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>

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

}  // namespace inte_vpux
}  // namespace ov
