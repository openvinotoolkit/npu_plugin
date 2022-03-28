//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once
#include <vector>
#include "vpux_driver_compiler_adapter.h"

namespace vpux {
namespace driverCompilerAdapter {

/**
 * @brief Contain all required transformation on nGraph function in case for external compiler usage and
 *  providing forward compatibility (Ngraph function with opset N+M, external compiler with opset N)
 */
namespace ngraphTransformations {

/**
 * @brief Check, can we compile and run ngraph function, if only specific opset version supported
 * @param opsetVersion Version of opset, which is supported
 */
bool isFunctionSupported(const std::shared_ptr<const ngraph::Function>& netGraph, size_t opsetVersion);

/**
 * @brief Serialize ngraph function to IR
 */
IR serializeToIR(const std::shared_ptr<ngraph::Function>& netGraph);

}  // namespace ngraphTransformations
}  // namespace driverCompilerAdapter
}  // namespace vpux
