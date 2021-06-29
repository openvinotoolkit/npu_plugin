// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSplitLayerTest : public SplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            // [Track number: E#9525]
            throw LayerTestsUtils::KmbSkipTestException("Issue in LoadNetwork() with blob from mcmCompiler, wasn't able to create NnCorePlg: 1");
        }
    }

    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Issues with Runtime. Outputs is empty because runtime doesn't wait while dma is finished");
    }
};

TEST_P(KmbSplitLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbSplitLayerTest, CompareWithRefs_MCM) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_Split, KmbSplitLayerTest,
                        ::testing::Combine(::testing::Values(2, 3), ::testing::Values(0, 1, 2, 3),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                                           ::testing::Values(InferenceEngine::Layout::NCHW,InferenceEngine::Layout::NHWC),
                                           ::testing::Values(InferenceEngine::Layout::NCHW,InferenceEngine::Layout::NHWC),
                                           ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                                           ::testing::Values(InferenceEngine::SizeVector({})),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        SplitLayerTest::getTestCaseName);
}  // namespace
