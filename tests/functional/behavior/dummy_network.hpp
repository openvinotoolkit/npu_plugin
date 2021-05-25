// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>

namespace {

InferenceEngine::CNNNetwork createDummyNetwork() {
    InferenceEngine::SizeVector inputShape = {1, 3, 4, 3};
    InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;
    size_t axis = 1;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);

    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};

    auto function = std::make_shared<ngraph::Function>(results, params, "softMax");
    InferenceEngine::CNNNetwork cnnNet(function);

    return cnnNet;
}

}