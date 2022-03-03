// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/adaptive_pooling.hpp"

namespace LayerTestsDefinitions {

class KmbAdaPoolLayerTest : public AdaPoolLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        std::string poolingMode;
        std::tie(std::ignore, std::ignore, poolingMode, std::ignore, targetDevice) = this->GetParam();
        if (poolingMode == "max") {
            throw LayerTestsUtils::KmbSkipTestException("MAX mode is unsupported for now");
        }
    }
};

TEST_P(KmbAdaPoolLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,

};

const auto AdaPoolAvg3DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 7}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{3}}),           // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}),              // mode
                           ::testing::ValuesIn(netPrecisions),                                // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));     // device

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPoolAvg3D, KmbAdaPoolLayerTest, AdaPoolAvg3DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);

const auto AdaPoolAvg4DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 32, 32}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{16, 16}}),           // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}),                   // mode
                           ::testing::ValuesIn(netPrecisions),                                     // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));          // device
INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPoolAvg4D, KmbAdaPoolLayerTest, AdaPoolAvg4DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);

const auto AdaPoolAvg5DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 17, 4, 5, 4}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{3, 5, 3}}),  // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"avg"}),           // mode
                           ::testing::ValuesIn(netPrecisions),                             // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));  // device
INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPoolAvg5D, KmbAdaPoolLayerTest, AdaPoolAvg5DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);

const auto AdaPoolMax3DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2, 3}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}}),           // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"max"}),              // mode
                           ::testing::ValuesIn(netPrecisions),                                // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));     // device
INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPoolMax3D, KmbAdaPoolLayerTest, AdaPoolMax3DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);
