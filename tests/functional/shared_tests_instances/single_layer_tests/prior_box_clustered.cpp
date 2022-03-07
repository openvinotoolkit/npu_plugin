// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/prior_box_clustered.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPriorBoxClusteredLayerTest :
        public PriorBoxClusteredLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbPriorBoxClusteredLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<float>> widths = {{5.12f, 14.6f, 13.5f}, {7.0f, 8.2f, 33.39f}};

const std::vector<std::vector<float>> heights = {{15.12f, 15.6f, 23.5f}, {10.0f, 16.2f, 36.2f}};

const std::vector<float> step_widths = {2.0f};

const std::vector<float> step_heights = {1.5f};

const std::vector<float> step = {1.5f};

const std::vector<float> offsets = {0.5f};

const std::vector<std::vector<float>> variances = {
        {0.1f, 0.1f, 0.2f, 0.2f},
};

const std::vector<bool> clips = {true, false};

const auto layerSpeficParams =
        testing::Combine(testing::ValuesIn(widths), testing::ValuesIn(heights), testing::ValuesIn(clips),
                         testing::ValuesIn(step_widths), testing::ValuesIn(step_heights), testing::ValuesIn(step),
                         testing::ValuesIn(offsets), testing::ValuesIn(variances));

INSTANTIATE_TEST_CASE_P(smoke_PriorBoxClustered, KmbPriorBoxClusteredLayerTest,
                        testing::Combine(layerSpeficParams, testing::Values(InferenceEngine::Precision::FP16),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Layout::ANY),
                                         testing::Values(InferenceEngine::Layout::ANY),
                                         testing::Values(std::vector<size_t>({4, 4})),
                                         testing::Values(std::vector<size_t>({50, 50})),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPriorBoxClusteredLayerTest::getTestCaseName);

}  // namespace
