//
// Copyright Intel Corporation.
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
