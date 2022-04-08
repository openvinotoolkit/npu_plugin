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

#include "vpux_compiler_l0_tests_common.h"

namespace VpuxCompilerL0TestsUtils {
std::shared_ptr<ngraph::Function> VpuxCompilerL0TestsCommon::create_simple_function() {
    // This example shows how to create ngraph::Function
    //
    // Parameter--->Multiply--->Add--->Result
    //    Constant---'          /
    //              Constant---'

    // Create opset3::Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f16, ngraph::Shape{3, 2});

    auto mul_constant = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {1.5});
    auto mul = std::make_shared<ngraph::opset3::Multiply>(data, mul_constant);

    auto add_constant = ngraph::opset3::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {0.5});
    auto add = std::make_shared<ngraph::opset3::Add>(mul, add_constant);

    // Create opset3::Result operation
    auto res = std::make_shared<ngraph::opset3::Result>(mul);

    // Create nGraph function
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
}

std::string VpuxCompilerL0TestsCommon::getTestModelsBasePath() {
    if (const auto envVar = std::getenv("POR_PATH")) {
        return envVar;
    }
    return {};
}
}  // namespace VpuxCompilerL0TestsUtils

