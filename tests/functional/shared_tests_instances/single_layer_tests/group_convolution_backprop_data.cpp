// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/group_convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

class KmbGroupConvBackpropDataLayerTest :
        public GroupConvBackpropDataLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeInfer() override {
        if (envConfig.IE_KMB_TESTS_PLATFORM == "3900") {
            throw LayerTestsUtils::KmbSkipTestException("Unsupported platform: 3900");
        }
    }
    void SkipBeforeLoad() override {
        groupConvBackpropDataSpecificParams groupConvBackpropDataParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InferenceEngine::SizeVector inputShapes;
        std::string targetDevice;
        std::tie(groupConvBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = GetParam();

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvBackpropDataParams;

        if (convOutChannels != numGroups || convOutChannels != inputShapes[1]) {
            throw LayerTestsUtils::KmbSkipTestException("Support only same number for groups, input and output channels");
        }
        if (padBegin != padEnd) {
            throw LayerTestsUtils::KmbSkipTestException("Support only symmetrical paddings");
        }
    }
};

TEST_P(KmbGroupConvBackpropDataLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

/* ============= 2D GroupConvBackpropData ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {1, 1}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {1, 1}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {12};
const std::vector<size_t> numGroups = {12};
const auto inputShapes = std::vector<std::vector<size_t>>({{1, 12, 7, 7}, {1, 12, 4, 4}});

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels), ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData2D_ExplicitPadding_NCHW, KmbGroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::NCHW),
                                           ::testing::Values(InferenceEngine::Layout::NCHW),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData2D_AutoPadValid_NCHW, KmbGroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::NCHW),
                                           ::testing::Values(InferenceEngine::Layout::NCHW),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData2D_ExplicitPadding_NHWC, KmbGroupConvBackpropDataLayerTest,
::testing::Combine(groupConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    ::testing::Values(InferenceEngine::Layout::NHWC),
                    ::testing::Values(InferenceEngine::Layout::NHWC),
                    ::testing::ValuesIn(inputShapes),
                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
KmbGroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData2D_AutoPadValid_NHWC, KmbGroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::NHWC),
                                           ::testing::Values(InferenceEngine::Layout::NHWC),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvBackpropDataLayerTest::getTestCaseName);
}  // namespace
