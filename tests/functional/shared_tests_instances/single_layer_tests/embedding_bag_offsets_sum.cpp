
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/embedding_bag_offsets_sum.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXEmbeddingBagOffsetsSumLayerTest :
        public EmbeddingBagOffsetsSumLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXEmbeddingBagOffsetsSumLayerTest_VPU3700 : public VPUXEmbeddingBagOffsetsSumLayerTest {};

TEST_P(VPUXEmbeddingBagOffsetsSumLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
};

const std::vector<InferenceEngine::Precision> indPrecisions = {
        InferenceEngine::Precision::I32,
};

const std::vector<std::vector<size_t>> emb_table_shape = {{10, 35, 8}, {5, 6}};
const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}};
const std::vector<std::vector<size_t>> offsets = {{0, 2}};
const std::vector<size_t> default_index = {0};
const std::vector<bool> with_weights = {true, false};
const std::vector<bool> with_default_index = {true, false};

const auto EmbeddingBagOffsetsSumParams1 = ::testing::Combine(
        ::testing::ValuesIn(emb_table_shape), ::testing::ValuesIn(indices), ::testing::ValuesIn(offsets),
        ::testing::ValuesIn(default_index), ::testing::ValuesIn(with_weights), ::testing::ValuesIn(with_default_index));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_EmbeddingBagOffsetsSum1, VPUXEmbeddingBagOffsetsSumLayerTest_VPU3700,
                        ::testing::Combine(EmbeddingBagOffsetsSumParams1, ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(indPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXEmbeddingBagOffsetsSumLayerTest_VPU3700::getTestCaseName);

}  // namespace
