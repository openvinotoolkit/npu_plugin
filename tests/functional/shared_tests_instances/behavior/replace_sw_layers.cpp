//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <base/behavior_test_utils.hpp>
#include <functional_test_utils/precision_utils.hpp>
#include <ie_core.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_metrics.hpp"

using CompileWithDummy = BehaviorTestsUtils::BehaviorTestsBasic;

InferenceEngine::CNNNetwork buildSingleLayerClampNetwork() {  // Clamp is not supported in SW
    InferenceEngine::SizeVector inputShape = {1, 3, 4, 3};
    InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto clamp = std::make_shared<ngraph::opset3::Clamp>(paramOuts.at(0), 0., 1.);

    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clamp)};

    auto function = std::make_shared<ngraph::Function>(results, params, "clamp");
    // Create CNNNetwork from ngraph::Function

    return InferenceEngine::CNNNetwork(function);
}

namespace {

TEST_P(CompileWithDummy, CompilationForSpecificPlatform) {
    if (getBackendName(*ie) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerClampNetwork();
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, target_device, configuration));
    }
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {
        {{VPUX_CONFIG_KEY(PLATFORM), "3720"}, {"VPUX_COMPILATION_MODE_PARAMS", "dummy-op-replacement=true"}}};
// Must be successfully compiled with dummy-op-replacement=true

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DummyVPU3720, CompileWithDummy,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         CompileWithDummy::getTestCaseName);
}  // namespace
