// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functions.h"
#include <functional_test_utils/precision_utils.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph_functions/builders.hpp>

InferenceEngine::CNNNetwork buildSingleLayerSoftMaxNetwork() {
	InferenceEngine::SizeVector inputShape = {1, 3, 4, 3};
	InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;
	size_t axis = 1;

	const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

	const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

	const auto paramOuts = ngraph::helpers::convert2OutputVector(
			ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

	const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);

	const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};

	auto function = std::make_shared<ngraph::Function>(results, params, "softMax");
	// Create CNNNetwork from ngraph::Function

	return InferenceEngine::CNNNetwork(function);
}

const std::string PlatformEnvironment::PLATFORM = []() -> std::string {
        if (const auto var = std::getenv("IE_KMB_TESTS_PLATFORM")) {
            return var;
        }

        return std::string("VPU3700");
}();
