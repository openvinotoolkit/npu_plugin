117 lines (97 sloc)  4.77 KB
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roll.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbRollTest: public RollTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbRollTest, CompareWithRefs_MLIR) {
   useCompilerMLIR();
   Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<InferenceEngine::Precision> inputPrecisions = {
            InferenceEngine::Precision::I16
    };

/*
const auto testCaseNegativeUnorderedAxes4D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{3, 11, 6, 4}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{7, 3}), // Shift
    ::testing::Values(std::vector<int64_t>{-3, -2}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);
*/

   const std::vector<InferenceEngine::SizeVector> inputShapes = {
           InferenceEngine::SizeVector {3, 11, 6, 4}
   };

   std::vector<int64_t> shift = {7,3};        // Shift
   std::vector<int64_t> axes = {-3, -2};        // Axes

    const auto testRollParams = ::testing::Combine(
           ::testing::ValuesIn(inputShapes),
           ::testing::ValuesIn(shift),
           ::testing::ValuesIn(axes),
           ::testing::ValuesIn(inputPrecisions),
           ::testing::ValuesIn(netPrecisions),
           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_CASE_P(smoke_Roll_Test,
                            KmbRollTest,
                            testRollParams,
                            KmbRollTest::getTestCaseName);

}  // namespace
