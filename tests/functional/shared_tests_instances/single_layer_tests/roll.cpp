// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roll.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbRollLayerTest: public RollLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbRollLayerTest, CompareWithRefs_MLIR) {
   useCompilerMLIR();
   Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

    /* OPENVINO
    typedef std::tuple<
        InferenceEngine::SizeVector, // Input shapes
        InferenceEngine::Precision,  // Input precision
        std::vector<int64_t>,        // Shift
        std::vector<int64_t>,        // Axes
        std::string> rollParams;   // Device name
    */

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<InferenceEngine::Precision> inputPrecisions = {
                InferenceEngine::Precision::I16,
                InferenceEngine::Precision::I32
    };


    std::vector<std::vector<size_t>> inputShapes = {
        {16}
    };
    
    const std::vector<std::vector<int64_t>> shift = { {5} };

    const std::vector<std::vector<int64_t>> axes = { {0} };

//     const std::vector<int64_t> shift =  {5} ;

//     const std::vector<int64_t> axes =  {0} ;

    const auto testRollParams = ::testing::Combine(
           ::testing::Values(inputShapes[0]),
           ::testing::ValuesIn(inputPrecisions),
           ::testing::ValuesIn(shift),
           ::testing::ValuesIn(axes),
           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_CASE_P(smoke_Roll_Test,
                            KmbRollLayerTest,
                            testRollParams,
                            KmbRollLayerTest::getTestCaseName);

}  // namespace
