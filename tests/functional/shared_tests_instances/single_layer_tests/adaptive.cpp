// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/adaptive_pooling.hpp"

namespace LayerTestsDefinitions {

class KmbAdaPoolLayerTest : public AdaPoolLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

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

const auto AdaPool3DCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 7}, {1, 1, 3}}),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),   // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),    // mode
                           ::testing::ValuesIn(netPrecisions),                             // precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));  // device

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TestsAdaPool3D, KmbAdaPoolLayerTest, AdaPool3DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);

const auto AdaPool4DCases = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 32, 32}, {1, 1, 3, 2}}),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{3, 5}, {16, 16}}),                 // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                          // mode
        ::testing::ValuesIn(netPrecisions),                                                   // precision
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));                        // device

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TestsAdaPool4D, KmbAdaPoolLayerTest, AdaPool4DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);

const auto AdaPool5DCases = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 17, 4, 5, 4}, {1, 1, 3, 2, 3}}),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 1}, {3, 5, 3}}),                  // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                               // mode
        ::testing::ValuesIn(netPrecisions),                                                        // precision
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));                             // device

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_TestsAdaPool5D, KmbAdaPoolLayerTest, AdaPool5DCases,
                        KmbAdaPoolLayerTest::getTestCaseName);
