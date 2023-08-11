//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

std::string VpuxCompilerL0TestsCommon::getCidToolPath() {
    if (const auto envVar = std::getenv("CID_TOOL")) {
        return envVar;
    }
    std::cerr << "CID_TOOL empty! You need to export CID_TOOL to load config!" << std::endl;
    return {};
}

void VpuxCompilerL0TestsCommon::postProcessNetOptions(const std::string& device) {
    netOptions += std::string{" VPUX_PLATFORM=\""} + device + std::string{"\""};
    netOptions += std::string{" DEVICE_ID=\""} + std::string{"VPUX."} + device + std::string{"\""};
}

std::string VpuxCompilerL0TestsCommon::getNetOptions() {
    return netOptions;
}

IRInfoTestType VpuxCompilerL0TestsCommon::readJson2Vec(std::string fileName) {
    IRInfoTestType ir_infos;
    std::ifstream i(fileName);
    std::string line;
    while (std::getline(i, line)) {
        std::unordered_map<std::string, std::string> ir_info;
        auto res = VpuxCompilerL0TestsUtils::Json::parse(line);
        for (auto it = res.begin(); it != res.end(); ++it) {
            ir_info.emplace(it.key(), it.value());
        }

        bool enabled;
        auto find = ir_info.find("enabled");
        std::istringstream enabled_is;
        if (find != ir_info.end()) {
            enabled_is.str(find->second);
        } else {
            std::cout << "Not found enabled entry, using false as its default\n";
            enabled_is.str("false");
        }
        enabled_is >> std::boolalpha >> enabled;

        if (enabled) {
            ir_infos.push_back(ir_info);
        }
    }
    return ir_infos;
}

}  // namespace VpuxCompilerL0TestsUtils
