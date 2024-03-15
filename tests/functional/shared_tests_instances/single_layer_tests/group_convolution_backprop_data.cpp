// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/group_convolution_backprop_data.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class GroupConvBackpropLayerTestCommon :
        public GroupConvBackpropLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class GroupConvBackpropLayerTest_NPU3700 : public GroupConvBackpropLayerTestCommon {};
class GroupConvBackpropLayerTest_NPU3720 : public GroupConvBackpropLayerTestCommon {};

TEST_P(GroupConvBackpropLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(GroupConvBackpropLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<size_t> numOutChannels = {64};
const std::vector<size_t> numGroups = {64};
const std::vector<std::vector<size_t>> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes2D = {{1, 64, 64, 64}};
const std::vector<std::vector<size_t>> kernels2D = {{4, 4}};
const std::vector<std::vector<size_t>> strides2D = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> outputPadding2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding));

const auto groupConvBackpropData2DParams_OutputPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding2D));

// ------ NPU3700 ------
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropLayerTest_NPU3700,
                         ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes2D), ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GroupConvBackpropLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_OutputPadding, GroupConvBackpropLayerTest_NPU3700,
                         ::testing::Combine(groupConvBackpropData2DParams_OutputPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes2D), ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GroupConvBackpropLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720 ------
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropLayerTest_NPU3720,
                         ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes2D), ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GroupConvBackpropLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_OutputPadding, GroupConvBackpropLayerTest_NPU3720,
                         ::testing::Combine(groupConvBackpropData2DParams_OutputPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes2D), ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GroupConvBackpropLayerTest_NPU3720::getTestCaseName);

}  // namespace
