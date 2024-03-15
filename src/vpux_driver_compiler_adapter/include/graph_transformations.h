//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <openvino/pass/manager.hpp>
#include <vector>
#include "vpux_driver_compiler_adapter.h"

namespace vpux {
namespace driverCompilerAdapter {

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace graphTransformations {

/**
 * @brief Check, can we compile and run OpenVINO model, if only specific opset version supported
 * @param opsetVersion Version of opset, which is supported
 */
bool isFunctionSupported(const std::shared_ptr<const ov::Model>& model, std::string opsetVersion);

/**
 * @brief Serialize OpenVINO model to IR
 */
IR serializeToIR(std::shared_ptr<ov::Model>& model, const uint32_t& supportedVersionByCompiler = 7);

void downgradeOpset(ov::pass::Manager& manager, const int& supportedVersionByCompiler);

}  // namespace graphTransformations
}  // namespace driverCompilerAdapter
}  // namespace vpux
