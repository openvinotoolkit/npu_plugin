// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/prior_box.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPriorBoxLayerTest : public PriorBoxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbPriorBoxLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const priorBoxSpecificParams param1 = {
        //(openvino eg)
        std::vector<float>{16.0},                // min_size
        std::vector<float>{38.46},               // max_size
        std::vector<float>{2.0},                 // aspect_ratio
        std::vector<float>{},                    // [density]
        std::vector<float>{},                    // [fixed_ratio]
        std::vector<float>{},                    // [fixed_size]
        false,                                   // clip
        true,                                    // flip
        16.0,                                    // step
        0.5,                                     // offset
        std::vector<float>{0.1, 0.1, 0.2, 0.2},  // variance
        false,                                   // [scale_all_sizes]
        false                                    // min_max_aspect_ratios_order ?
};

const priorBoxSpecificParams param2 = {
        std::vector<float>{2.0},  // min_size
        std::vector<float>{5.0},  // max_size
        std::vector<float>{1.5},  // aspect_ratio
        std::vector<float>{},     // [density]
        std::vector<float>{},     // [fixed_ratio]
        std::vector<float>{},     // [fixed_size]
        false,                    // clip
        false,                    // flip
        1.0,                      // step
        0.0,                      // offset
        std::vector<float>{},     // variance
        false,                    // [scale_all_sizes]
        false                     // min_max_aspect_ratios_order
};

const priorBoxSpecificParams param3 = {
        std::vector<float>{256.0},  // min_size
        std::vector<float>{315.0},  // max_size
        std::vector<float>{2.0},    // aspect_ratio
        std::vector<float>{},       // [density]
        std::vector<float>{},       // [fixed_ratio]
        std::vector<float>{},       // [fixed_size]
        true,                       // clip
        true,                       // flip
        1.0,                        // step
        0.0,                        // offset
        std::vector<float>{},       // variance
        true,                       // [scale_all_sizes]
        false                       // min_max_aspect_ratios_order
};

const InferenceEngine::SizeVector inputShape1 = {24, 42};
const InferenceEngine::SizeVector inputShape2 = {2, 2};
const InferenceEngine::SizeVector inputShape3 = {1, 1};

const InferenceEngine::SizeVector imageShape1 = {348, 672};
const InferenceEngine::SizeVector imageShape2 = {10, 10};
const InferenceEngine::SizeVector imageShape3 = {300, 300};

INSTANTIATE_TEST_CASE_P(smoke_PriorBox_1, KmbPriorBoxLayerTest,
                        testing::Combine(testing::Values(param1), testing::Values(InferenceEngine::Precision::FP16),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Layout::ANY),
                                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(inputShape1),
                                         testing::Values(imageShape1),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPriorBoxLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_PriorBox_2, KmbPriorBoxLayerTest,
                        testing::Combine(testing::Values(param2), testing::Values(InferenceEngine::Precision::FP16),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Layout::ANY),
                                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(inputShape2),
                                         testing::Values(imageShape2),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPriorBoxLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_PriorBox_3, KmbPriorBoxLayerTest,
                        testing::Combine(testing::Values(param3), testing::Values(InferenceEngine::Precision::FP16),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                         testing::Values(InferenceEngine::Layout::ANY),
                                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(inputShape3),
                                         testing::Values(imageShape3),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPriorBoxLayerTest::getTestCaseName);

}  // namespace
