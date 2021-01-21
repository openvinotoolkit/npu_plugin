// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

std::set<ngraph::helpers::ActivationTypes> supportedTypes {
    ngraph::helpers::Relu,
    ngraph::helpers::Sigmoid,
    ngraph::helpers::HSwish,
    ngraph::helpers::Tanh,
    ngraph::helpers::SoftPlus
};

std::set<ngraph::helpers::ActivationTypes> supportedTypesByExperimentalCompiler {
    ngraph::helpers::Relu,
    ngraph::helpers::Sigmoid,
    ngraph::helpers::Clamp,
    ngraph::helpers::Elu,
    ngraph::helpers::HSwish,
    ngraph::helpers::Tanh,
    ngraph::helpers::PReLu,
    ngraph::helpers::LeakyRelu
};

class KmbActivationLayerTest : public ActivationLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationParam;
        std::tie(activationParam,
                 std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore, std::ignore, std::ignore) = GetParam();

        const auto activationType = activationParam.first;

        if (!envConfig.IE_VPUX_USE_EXPERIMENTAL_COMPILER) {
            if (supportedTypes.find(activationType) ==
                supportedTypes.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Unsupported activation types in MCM compiler");
            }
        } else {
            if (supportedTypesByExperimentalCompiler.find(activationType) ==
                supportedTypesByExperimentalCompiler.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Experimental compiler doesn't supports activation type " +
                                                            LayerTestsDefinitions::activationNames[activationType] +
                                                            " yet");
            }
        }
    }
};

TEST_P(KmbActivationLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid,  {{1.0f}}},
    {Tanh,     {{1.0f}}},
    {Relu,     {{1.0f}}},
    {Exp,      {{1.0f}}},
    {Log,      {{1.0f}}},
    {Sign,     {{1.0f}}},
    {Abs,      {{1.0f}}},
    {Elu,      {{1.0f}}},
    {Clamp,    {{-1.0f, 1.0f}}},
    {HSwish,   {{1.0f}}},
    {PReLu,    {{0.01f}}},
    {LeakyRelu,{{0.01f}}},
    {SoftPlus, {{1.0f}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50, 1, 1}, {{1}, {50}}},
    {{1, 128, 1, 1}, {{1}, {128}}},
};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Activation_Test, KmbActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
