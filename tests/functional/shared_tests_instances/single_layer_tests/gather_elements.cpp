// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/gather_elements.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {

class KmbGatherElementsLayerTest: public GatherElementsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            if (envConfig.IE_KMB_TESTS_RUN_INFER) {
                throw LayerTestsUtils::KmbSkipTestException("Issues with MCM compiler");
            }
        }
    }
    void SkipBeforeInfer() override {
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::KmbSkipTestException("Bad results on Level0");
        }
    }

};

TEST_P(KmbGatherElementsLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbGatherElementsLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> dPrecisions = {
            InferenceEngine::Precision::FP32
    };

    const std::vector<InferenceEngine::Precision> iPrecisions = {
            InferenceEngine::Precision::I32
    };

    const std::vector<int> axes_set1 = {-1,0,1};
    const std::vector<int> axes_set2 = {-2,1};
    const std::vector<int> axes_set3 = {0};

    INSTANTIATE_TEST_SUITE_P(
            smoke_GatherElements_set1,
            KmbGatherElementsLayerTest,
            testing::Combine(
                testing::Values(std::vector<size_t>{2,2}),
                testing::Values(std::vector<size_t>{2,2}),
                testing::ValuesIn(axes_set1),
                testing::ValuesIn(dPrecisions),
                testing::ValuesIn(iPrecisions),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbGatherElementsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_GatherElements_set2,
            KmbGatherElementsLayerTest,
            testing::Combine(
                testing::Values(std::vector<size_t>{5,7,9,1}),
                testing::Values(std::vector<size_t>{5,7,9,1}),
                testing::ValuesIn(axes_set2),
                testing::ValuesIn(dPrecisions),
                testing::ValuesIn(iPrecisions),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbGatherElementsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_GatherElements_set3,
            KmbGatherElementsLayerTest,
            testing::Combine(
                testing::Values(std::vector<size_t>{2,2,1}),
                testing::Values(std::vector<size_t>{4,2,1}),
                testing::ValuesIn(axes_set3),
                testing::ValuesIn(dPrecisions),
                testing::ValuesIn(iPrecisions),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbGatherElementsLayerTest::getTestCaseName
    );
}  // namespace
