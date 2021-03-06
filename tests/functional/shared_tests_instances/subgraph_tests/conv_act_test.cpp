// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "single_layer_tests/convolution.hpp"
#include "conv_act_base.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace SubgraphTestsDefinitions {

class KmbConvActivationSubgraphTest : public ConvActTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};;

TEST_P(KmbConvActivationSubgraphTest, CompareWithRefs) {
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
const std::vector<size_t> numOutCannels = {64};
const std::vector<ngraph::op::PadType> padTypes = {ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<InferenceEngine::Precision> outputPrecisions = {
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Mish,         {{}}},
        {LeakyRelu,    {{0.1f}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 64, 1, 1}, {{}}},
        //{{1, 50, 1, 1}, {{}}}, - error in fp16 network
        {{1, 128, 1, 1}, {{}}}, // should cover most of u8 values
};

const auto activationCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(outputPrecisions),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));


const auto convCases = ::testing::Combine(activationCases,
                                          ::testing::ValuesIn(kernels),
                                          ::testing::ValuesIn(strides),
                                          ::testing::ValuesIn(padBegins),
                                          ::testing::ValuesIn(padEnds),
                                          ::testing::ValuesIn(dilations),
                                          ::testing::ValuesIn(numOutCannels),
                                          ::testing::Values(ngraph::op::PadType::EXPLICIT));


INSTANTIATE_TEST_CASE_P(smoke_ConvActivation_Test,
                        KmbConvActivationSubgraphTest,
                        convCases,
                        ConvActTest::getTestCaseName);


}  // namespace

