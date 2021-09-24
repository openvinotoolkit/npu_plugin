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
            throw LayerTestsUtils::KmbSkipTestException("CallVpu error: -1");
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

/* ============= 2D GroupDeconv ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {1, 1}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {12};
const std::vector<size_t> numGroups = {12};
const auto inputShapes = std::vector<size_t>({1, 12, 7, 7});

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels), ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv2D_ExplicitPadding, KmbGroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(std::vector<size_t>(inputShapes)),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv2D_AutoPadValid, KmbGroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(inputShapes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbGroupConvBackpropDataLayerTest::getTestCaseName);
}  // namespace
