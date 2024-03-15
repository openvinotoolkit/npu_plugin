// Copyright (C) 2019-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "conv_act_base.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace SubgraphTestsDefinitions {
class ConvActivationSubgraphTestCommon : public ConvActTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class ConvActivationSubgraphTest_NPU3700 : public ConvActivationSubgraphTestCommon {};
class ConvActivationSubgraphTest_NPU3720 : public ConvActivationSubgraphTestCommon {};

TEST_P(ConvActivationSubgraphTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ConvActivationSubgraphTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace SubgraphTestsDefinitions

using namespace SubgraphTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::U8,
};

const InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
const InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;

/* ============= 2D Convolution ============= */

const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{1, 1}};
const std::vector<InferenceEngine::SizeVector> dilations = {{1, 1}};

/* ============= 3D Convolution ============= */

const std::vector<InferenceEngine::SizeVector> kernels3D = {{3, 3, 3}};
const std::vector<InferenceEngine::SizeVector> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{1, 1, 1}};
const std::vector<InferenceEngine::SizeVector> dilations3D = {{1, 1, 1}};

const std::vector<size_t> numOutCannels = {64};
const std::vector<ngraph::op::PadType> padTypes = {ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID};

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<InferenceEngine::Precision> outputPrecisions = {InferenceEngine::Precision::FP16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Mish, {{}}},
        {LeakyRelu, {{0.1f}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 64, 1, 1}, {{}}},
        //{{1, 50, 1, 1}, {{}}}, - error in fp16 network
        {{1, 128, 1, 1}, {{}}},  // should cover most of u8 values
};

const auto activationCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)), ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(outputPrecisions),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto convCases =
        ::testing::Combine(activationCases, ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
                           ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
                           ::testing::ValuesIn(numOutCannels), ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv3DCases = ::testing::Combine(
        activationCases, ::testing::ValuesIn(kernels3D), ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D), ::testing::ValuesIn(padEnds3D), ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutCannels), ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(smoke_ConvActivation_Test, ConvActivationSubgraphTest_NPU3700, convCases,
                         ConvActTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvActivation_Test, ConvActivationSubgraphTest_NPU3720, convCases,
                         ConvActTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_3DConvActivation_Test, ConvActivationSubgraphTest_NPU3720, conv3DCases,
                         ConvActTest::getTestCaseName);

}  // namespace
