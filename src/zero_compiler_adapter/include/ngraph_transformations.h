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
#include "zero_compiler_adapter.h"

namespace vpux {
namespace zeroCompilerAdapter {

/**
 * @brief Contain all required transformation on nGraph function
 */
namespace ngraphTransformations {

/**
 * @brief Perform lowering if it's required.
 * @warning If conversion is not possible, this function will just skip this step
 */
void applyLoweringPasses(const std::shared_ptr<ngraph::Function>& netGraph, Opset opsetVersion);

/**
 * @brief Check, can we compile and run ngraph function, if only specific opset version supported
 * @param opsetVersion Version of opset, which is supported
 */
bool isFunctionSupported(const std::shared_ptr<const ngraph::Function>& netGraph, Opset opsetVersion);

/**
 * @brief Serialize ngraph function to IR
 */
IR serializeToIR(const std::shared_ptr<ngraph::Function>& netGraph);

// TODO Only MVN6 operation included right now
/**
 * @brief Apply ops lowering from opset6, where it's possible
 */
void lowerFromOpset6(const std::shared_ptr<ngraph::Function>& netGraph);

}  // namespace ngraphTransformations
}  // namespace zeroCompilerAdapter
}  // namespace vpux
