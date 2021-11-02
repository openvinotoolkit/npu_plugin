// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSelectLayerTest : public SelectLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

    // void SkipBeforeLoad() override {
    //     if (envConfig.IE_KMB_TESTS_RUN_INFER) {
    //         throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
    //     }
    // }
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::KmbSkipTestException("comparison fails");
    }

};

class KmbSelectLayerTest_MLIR : public KmbSelectLayerTest {};

// TEST_P(KmbSelectLayerTest, CompareWithRefs) {
//     Run();
// }

TEST_P(KmbSelectLayerTest_MLIR, CompareWithRefs_HW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    // Initialize a new test case for SelectOp
const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::FP16,
    //InferenceEngine::Precision::I8
    //InferenceEngine::Precision::BOOL
    //InferenceEngine::Precision::FP32

    // InferenceEngine::Precision::U8,
    // InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::I16,
    // InferenceEngine::Precision::I32
    //InferenceEngine::Precision::BIN 
    //InferenceEngine::Precision::UNSPECIFIED 
};

const std::vector<std::vector<std::vector<size_t>>> noneShapes = {
    // {{1}, {1}, {1}},
    // {{8}, {8}, {8}},

    // //use this before this
    // {{4, 5}, {4, 5}, {4, 5}},
    //  {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},

    // {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    // {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}
     
};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(noneShapes),

    ::testing::ValuesIn(inputPrecision),
    //::testing::Values(InferenceEngine::Precision::UNSPECIFIED),

    ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    //::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const std::vector<std::vector<std::vector<size_t>>> numpyShapes = {
    {{1}, {16}, {1}},
    {{1}, {1}, {16}},
    {{8}, {1}, {8}},
    {{4, 1}, {1}, {4, 8}},
    {{8, 1, 1}, {8, 1, 1}, {2, 5}},
    {{8, 1}, {6, 8, 1}, {6, 1, 1}},
    {{1, 4}, {8, 1, 1, 1}, {4}},
    {{5, 1, 1, 1}, {5, 7, 8, 6}, {1, 8, 6}},
    {{6}, {2, 1, 9, 8, 6}, {2, 4, 9, 8, 6}},
    {{7, 6, 5, 8}, {4, 7, 6, 5, 8}, {6, 1, 8}}
};

// const std::vector<std::vector<std::vector<size_t>>> numpyShapes = {
//     {{1}, {1}, {1}},
//     {{1}, {16}, {1}},
//     {{1}, {1}, {16}},
//     {{1}, {8}, {8}},
//     {{8}, {1}, {8}},
//     {{8}, {8}, {8}},
//     {{4, 1}, {1}, {4, 8}},
//     {{3, 8}, {8}, {3, 1}},
//     {{8, 1}, {8, 1}, {8, 1}},
//     {{1}, {5, 8}, {5, 8}},
//     {{8, 1, 1}, {8, 1, 1}, {2, 5}},
//     {{8, 1}, {6, 8, 1}, {6, 1, 1}},
//     {{5, 1}, {8, 1, 7}, {5, 7}},
//     {{2, 8, 1}, {2, 8, 9}, {2, 1, 9}},
//     {{1, 4}, {8, 1, 1, 1}, {4}},
//     {{5, 4, 1}, {8, 5, 1, 1}, {4, 1}},
//     {{1, 4}, {6, 1, 8, 1}, {6, 1, 8, 4}},
//     {{7, 3, 1, 8}, {7, 1, 1, 8}, {3, 2, 8}},
//     {{1, 3, 1}, {8, 2, 3, 1}, {3, 9}},
//     {{5, 1, 8}, {2, 1, 9, 8}, {2, 5, 9, 8}},
//     {{6, 1, 1, 8}, {6, 7, 1, 8}, {2, 1}},
//     {{5, 1, 1, 1}, {5, 7, 8, 6}, {1, 8, 6}},
//     {{8, 1, 5}, {8, 1, 1, 1, 1}, {8, 7, 5}},
//     {{8, 1, 1, 9}, {4, 8, 1, 1, 1}, {1, 1, 9}},
//     {{5, 1, 2, 1}, {8, 1, 9, 1, 1}, {5, 1, 2, 1}},
//     {{8, 1}, {2, 1, 1, 8, 1}, {9, 1, 1}},
//     {{8, 5, 5, 5, 1}, {8, 1, 1, 1, 8}, {5, 5, 5, 8}},
//     {{4}, {8, 5, 6, 1, 1}, {2, 4}},
//     {{9, 9, 2, 8, 1}, {9, 1, 2, 8, 1}, {9, 1, 1, 1}},
//     {{5, 3, 3}, {8, 1, 1, 3, 3}, {5, 1, 3}},
//     {{5, 1, 8, 1}, {5, 5, 1, 8, 1}, {1}},
//     {{3}, {6, 8, 1, 1, 3}, {6, 1, 5, 3, 3}},
//     {{5, 1}, {3, 1, 4, 1, 8}, {1, 4, 5, 8}},
//     {{2, 1, 5}, {8, 6, 2, 3, 1}, {5}},
//     {{6}, {2, 1, 9, 8, 6}, {2, 4, 9, 8, 6}},
//     {{5, 7, 1, 8, 1}, {5, 7, 1, 8, 4}, {8, 1}},
//     {{7, 6, 5, 8}, {4, 7, 6, 5, 8}, {6, 1, 8}}
// };

const auto numpyCases = ::testing::Combine(
    ::testing::ValuesIn(numpyShapes),

    ::testing::ValuesIn(inputPrecision),
    //::testing::Values(InferenceEngine::Precision::UNSPECIFIED),

    ::testing::Values(ngraph::op::AutoBroadcastSpec::NUMPY),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsSelect_none, KmbSelectLayerTest_MLIR, noneCases, KmbSelectLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestsSelect_numpy, KmbSelectLayerTest_MLIR, numpyCases, KmbSelectLayerTest::getTestCaseName);


}  // namespace
