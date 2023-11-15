//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/embedding_segments_sum.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXEmbeddingSegmentsSumLayerTest :
        public EmbeddingSegmentsSumLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXEmbeddingSegmentsSumLayerTest_VPU3700 : public VPUXEmbeddingSegmentsSumLayerTest {};
class VPUXEmbeddingSegmentsSumLayerTest_VPU3720 : public VPUXEmbeddingSegmentsSumLayerTest {};

TEST_P(VPUXEmbeddingSegmentsSumLayerTest_VPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXEmbeddingSegmentsSumLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::U8};

const std::vector<InferenceEngine::Precision> indPrecisions = {InferenceEngine::Precision::I32};

const std::vector<std::vector<size_t>> embTableShape = {{5, 6, 4}, {10, 35, 8}};
const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}};
const std::vector<std::vector<size_t>> segmentIds = {{0, 1, 2, 3, 4}};
const std::vector<size_t> numSegments = {7};
const std::vector<size_t> defaultIndex = {0};
const std::vector<bool> withWeights = {false, true};
const std::vector<bool> withDefaultIndex = {false, true};

const auto params = testing::Combine(::testing::ValuesIn(embTableShape), ::testing::ValuesIn(indices),
                                     ::testing::ValuesIn(segmentIds), ::testing::ValuesIn(numSegments),
                                     ::testing::ValuesIn(defaultIndex), ::testing::ValuesIn(withWeights),
                                     ::testing::ValuesIn(withDefaultIndex));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_EmbeddingSegmentsSumCheck1, VPUXEmbeddingSegmentsSumLayerTest_VPU3700,
                        ::testing::Combine(params, ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(indPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        VPUXEmbeddingSegmentsSumLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_EmbeddingSegmentsSumCheck1, VPUXEmbeddingSegmentsSumLayerTest_VPU3720,
                        ::testing::Combine(params, ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(indPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        VPUXEmbeddingSegmentsSumLayerTest_VPU3720::getTestCaseName);

}  // namespace
