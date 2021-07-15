// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/group_convolution.hpp"

namespace LayerTestsDefinitions {

class KmbGroupConvolutionLayerTest :
        public GroupConvolutionLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            const auto& params = std::get<0>(GetParam());
            const auto dilations = std::get<4>(params);
            if (dilations.size() == 1 && dilations[0] != 1) {
                throw LayerTestsUtils::KmbSkipTestException("Assert dilation.at(0) == dilation.at(1)");
            }
        }
    }

    // There is difference between actual and expected values during validation step on KMB-board:
    // KmbLayerTestsCommon::Validate()
    // LayerTestsCommon::Validate()
    // openvino/inference-engine/tests/functional/shared_test_classes/include
    // /shared_test_classes/base/layer_test_utils.hpp:173: Failure
    // Value of: max != 0 && (diff <= static_cast<float>(threshold))
    // Actual: false
    // Expected: true
    // Relative comparison of values expected: 446 and actual: 482 at index 0 with
    // threshold 0.0099999997764825821 failed
    // [Track number: S#50873]
    void SkipBeforeValidate() override {
        if (isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("Comparison fails");
        }

        if (isCompilerMLIR()) {
            auto params = std::get<0>(GetParam());

            auto strides = std::get<1>(params);
            auto padsBegin = std::get<2>(params);
            auto dilations = std::get<4>(params);
            auto padType = std::get<7>(params);

            // [Track number: E#16206]
            if (strides.size() == 1 && padType == ngraph::op::PadType::EXPLICIT) {
                auto isBadPadsBegin = padsBegin[0] > 1;
                auto isBadDilations = dilations[0] == 1;
                if (isBadPadsBegin && isBadDilations) {
                    throw LayerTestsUtils::KmbSkipTestException("Comparison fails");
                }
            }

            // [Track number: E#16206]
            if (strides.size() == 2 && padType == ngraph::op::PadType::EXPLICIT) {
                auto isBadStrides = strides[0] == strides[1] && strides[0] < 5;
                auto isGoodPadsBegin =
                        (padsBegin[0] == 0 && padsBegin[1] == 0) || (padsBegin[0] == 1 && padsBegin[1] == 0) ||
                        (padsBegin[0] == 0 && padsBegin[1] == 1) || (padsBegin[0] == 1 && padsBegin[1] == 1);
                auto isBadDilations = dilations[0] == 1 && dilations[1] == 1;

                if (isBadStrides && !isGoodPadsBegin && isBadDilations)
                    throw LayerTestsUtils::KmbSkipTestException("Comparison fails");
            }
        }
    }

    // [Track number: E#12804]
    void SkipBeforeInfer() override {
        if (envConfig.IE_KMB_TESTS_PLATFORM == "3900") {
            throw LayerTestsUtils::KmbSkipTestException("CallVpu error: -1");
        }
    }
};

TEST_P(KmbGroupConvolutionLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbGroupConvolutionLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbGroupConvolutionLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

/* ============= 1D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels1d = {{3}};
const std::vector<std::vector<size_t>> strides1d = {{1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1d = {{0}, {2}};
const std::vector<std::vector<ptrdiff_t>> padEnds1d = {{0}, {2}};
const std::vector<std::vector<size_t>> dilations1d = {{1}, {2}};
const std::vector<size_t> numOutChannels1d = {8, 16};
const std::vector<size_t> numGroups1d = {2, 8};
const auto inputShapes1d = std::vector<size_t>({1, 16, 30});

const auto groupConv1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d), ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d), ::testing::ValuesIn(dilations1d), ::testing::ValuesIn(numOutChannels1d),
        ::testing::ValuesIn(numGroups1d), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv1DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d), ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<ptrdiff_t>({0})), ::testing::ValuesIn(dilations1d),
        ::testing::ValuesIn(numOutChannels1d), ::testing::ValuesIn(numGroups1d),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolution1D_ExplicitPadding, KmbGroupConvolutionLayerTest,
                        ::testing::Combine(groupConv1DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>(inputShapes1d)),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolution1D_AutoPadValid, KmbGroupConvolutionLayerTest,
                        ::testing::Combine(groupConv1DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 16, 30})),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvolutionLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};
const auto inputShapes = std::vector<size_t>({1, 16, 30, 30});

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels), ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolution2D_ExplicitPadding, KmbGroupConvolutionLayerTest,
                        ::testing::Combine(groupConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>(inputShapes)),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvolution2D_AutoPadValid, KmbGroupConvolutionLayerTest,
                        ::testing::Combine(groupConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 16, 30, 30})),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvolutionLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}};
const auto inputShapes3d = std::vector<size_t>({1, 4, 10, 10, 10});

const auto groupConv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d), ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(dilations3d), ::testing::Values(4), ::testing::Values(2),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv3DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), ::testing::ValuesIn(dilations3d),
                           ::testing::Values(4), ::testing::Values(2), ::testing::Values(ngraph::op::PadType::VALID));

// Test is disabled because there is Segmentation fault (core dumped) at step
// [Debug  ][VPU][KMB nGraph Parser] Convert nGraph to MCM Model
// [Track number: S#50872]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_GroupConvolution3D_ExplicitPadding, KmbGroupConvolutionLayerTest,
                        ::testing::Combine(groupConv3DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>(inputShapes3d)),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvolutionLayerTest::getTestCaseName);

// Test is disabled because there is Segmentation fault (core dumped) at step
// [Debug  ][VPU][KMB nGraph Parser] Convert nGraph to MCM Model
// [Track number: S#50872]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_GroupConvolution3D_AutoPadValid, KmbGroupConvolutionLayerTest,
                        ::testing::Combine(groupConv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>({1, 4, 10, 10, 10})),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvolutionLayerTest::getTestCaseName);

}  // namespace
