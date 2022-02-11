// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/adaptive_pooling.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {

class KmbAdaPoolLayerTest : public AdaPoolLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
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

TEST_P(KmbAdaPoolLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

/* ============= Adaptive_AVG_Pool / 3D ============= */

     const std::vector<InferenceEngine::Precision> inputPrecisions = {
           // InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
        //    InferenceEngine::Precision::U8,
    };


const auto AdaPool3DCases =
        ::testing::Combine(::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        // { 1, 2, 1},
                        // { 1, 1, 3 },
                        // { 3, 17, 5 }}
                    {1, 3, 32, 32}
                }),

        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1}, {3}, {5} }),
        ::testing::ValuesIn(std::vector<std::string>{"avg"}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool3D, KmbAdaPoolLayerTest, AdaPool3DCases, KmbAdaPoolLayerTest::getTestCaseName);

