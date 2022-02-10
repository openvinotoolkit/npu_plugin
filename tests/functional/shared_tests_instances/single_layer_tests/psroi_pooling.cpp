// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/psroi_pooling.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {

class KmbPSROIPoolingLayerTest : public PSROIPoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            if (envConfig.IE_KMB_TESTS_RUN_INFER) {
                // [Track number: S#44493]
                // Test hangs on the the board
                throw LayerTestsUtils::KmbSkipTestException("Issues with MCM compiler");
            }
        }
    }
};

TEST_P(KmbPSROIPoolingLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapeVector0 = {
        { 2, 200, 20, 20 },
        { 2, 200, 20, 16 },
        { 2, 200, 16, 20 },
        { 3, 200, 16, 16 }
};
const std::vector<std::vector<size_t>> inputShapeVector1 = {
        { 1, 392, 14, 14 },
        { 1, 392, 38, 64 }
};
const std::vector<std::vector<size_t>> inputShapeVector2 = {
        {1, 49 * 1, 14, 14}
};

const std::vector<std::vector<size_t>> coordShapesVector0 = {
        { 1, 5 }
};
const std::vector<std::vector<size_t>> coordShapesVector1 = {
        { 300, 5 }
};


INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest0, KmbPSROIPoolingLayerTest,
                        testing::Combine(
                                ::testing::ValuesIn(inputShapeVector0),
                                ::testing::ValuesIn(coordShapesVector0),
                                ::testing::Values(50),
                                ::testing::Values(2),
                                ::testing::Values(1.0f),
                                ::testing::Values(1),
                                ::testing::Values(1),
                                ::testing::Values("average"),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPSROIPoolingLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest1, KmbPSROIPoolingLayerTest,
                        testing::Combine(
                                ::testing::ValuesIn(inputShapeVector1),
                                ::testing::ValuesIn(coordShapesVector1),
                                ::testing::Values(8),
                                ::testing::Values(7),
                                ::testing::Values(0.0625f),
                                ::testing::Values(1),
                                ::testing::Values(1),
                                ::testing::Values("average"),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPSROIPoolingLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest2, KmbPSROIPoolingLayerTest,
                        testing::Combine(
                                ::testing::ValuesIn(inputShapeVector2),
                                ::testing::ValuesIn(coordShapesVector0),
                                ::testing::Values(1),
                                ::testing::Values(7),
                                ::testing::Values(0.0625f),
                                ::testing::Values(1),
                                ::testing::Values(1),
                                ::testing::Values("average"),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbPSROIPoolingLayerTest::getTestCaseName);
