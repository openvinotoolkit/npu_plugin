//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/embedding_bag_packed_sum.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXEmbeddingBagPackedSumLayerTest_VPU3720 :
        public EmbeddingBagPackedSumLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(VPUXEmbeddingBagPackedSumLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<size_t> embTableShape = {5, 10};
const std::vector<std::vector<size_t>> indices = {{0, 2}, {1, 2}, {3, 4}};
const std::vector<bool> withWeights = {true, false};
const auto params = testing::Combine(::testing::Values(embTableShape), ::testing::Values(indices),
                                     ::testing::ValuesIn(withWeights));

const InferenceEngine::Precision embeddingTablePrecision = InferenceEngine::Precision::FP16;
const InferenceEngine::Precision indicesPrecisions = InferenceEngine::Precision::I32;

INSTANTIATE_TEST_SUITE_P(smoke_precommit_EmbeddingBagPackedSum_VPU3720, VPUXEmbeddingBagPackedSumLayerTest_VPU3720,
                         ::testing::Combine(params, ::testing::Values(embeddingTablePrecision),
                                            ::testing::Values(indicesPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         EmbeddingBagPackedSumLayerTest::getTestCaseName);

}  // namespace
