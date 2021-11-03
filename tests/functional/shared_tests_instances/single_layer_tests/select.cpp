// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbSelectLayerTest : public SelectLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeValidate() override {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    };

    class KmbSelectLayerTest_MLIR : public KmbSelectLayerTest {};

    TEST_P(KmbSelectLayerTest_MLIR, CompareWithRefs_HW) {
        useCompilerMLIR();
        setReferenceSoftwareModeMLIR();
        Run();
    }
} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecision = {
    //InferenceEngine::Precision::I8
    InferenceEngine::Precision::FP16,
    //InferenceEngine::Precision::FP32 
};

const std::vector<std::vector<std::vector<size_t>>> noneShapes = {
    {{1}, {1}, {1}},
    {{8}, {8}, {8}},
    {{4, 5}, {4, 5}, {4, 5}},
    {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}
};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(noneShapes),
    ::testing::ValuesIn(inputPrecision),

    ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const std::vector<std::vector<std::vector<size_t>>> numpyShapes = {
    {{1}, {1}, {1}},
    {{1}, {1}, {16}},
    {{1}, {8}, {8}},
    {{8}, {1}, {8}},
    {{8}, {8}, {8}},
    {{4, 1}, {1}, {4, 8}},
    {{8, 1}, {8, 1}, {8, 1}},
};

const auto numpyCases = ::testing::Combine(
    ::testing::ValuesIn(numpyShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NUMPY),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsSelectOp_none, KmbSelectLayerTest_MLIR, noneCases, KmbSelectLayerTest::getTestCaseName);

//// following test stop at the VPUIP_2 (more than 15min) at stage 
//// PIPE:LRT0       : W: [   5520893] application/demo/InferenceManagerDemo/leon/main.cpp:457     Network loading completed in 0.82 ms
// INSTANTIATE_TEST_CASE_P(smoke_TestsSelectOp_numpy, KmbSelectLayerTest_MLIR, numpyCases, KmbSelectLayerTest::getTestCaseName);


// const std::vector<std::vector<std::vector<size_t>>> numpyShapes2 = {
//     // 'async.yield' op operand types do not match the types returned from the parent ExecuteOp
//     {{1}, {16}, {1}},                            
//     {{3, 8}, {8}, {3, 1}},                       
//     {{2, 8, 1}, {2, 8, 9}, {2, 1, 9}},           

//     // Got non broadcastable dimensions pair : '4113' and 5'
//     // AdjustLayouts Pass failed : New order  doesn't match original rank 
//     {{1}, {5, 8}, {5, 8}},                       
//     {{8, 1}, {6, 8, 1}, {6, 1, 1}},              // AdjustLayouts Pass failed : New order 'NC' doesn't match original rank '3'
//     {{5, 1}, {8, 1, 7}, {5, 7}},                 // AdjustLayouts Pass failed : New order 'NC' doesn't match original rank '3'
//     {{1, 4}, {8, 1, 1, 1}, {4}},                 // AdjustLayouts Pass failed : New order 'NC' doesn't match original rank '4'
//     {{5, 4, 1}, {8, 5, 1, 1}, {4, 1}},           // AdjustLayouts Pass failed : New order 'CHW' doesn't match original rank '4' 
//     {{1, 4}, {6, 1, 8, 1}, {6, 1, 8, 4}},        // AdjustLayouts Pass failed : New order 'NC' doesn't match original rank '4'
//     {{1, 3, 1}, {8, 2, 3, 1}, {3, 9}},           // AdjustLayouts Pass failed : New order 'CHW' doesn't match original rank '4
//     {{5, 1, 8}, {2, 1, 9, 8}, {2, 5, 9, 8}},     // AdjustLayouts Pass failed : New order 'CHW' doesn't match original rank '4

//     //// Operation must have the same input and output order. inL=CHW, outL=NC
//     {{8, 1, 1}, {8, 1, 1}, {2, 5}},              
//     {{7, 3, 1, 8}, {7, 1, 1, 8}, {3, 2, 8}},     // Operation must have the same input and output order. inL=NCHW, outL=NC
//     {{6, 1, 1, 8}, {6, 7, 1, 8}, {2, 1}},        // Operation must have the same input and output order. inL=NCHW, outL=NC
//     {{5, 1, 1, 1}, {5, 7, 8, 6}, {1, 8, 6}},     // Operation must have the same input and output order. inL=NCHW, outL=CHW
// };

// const auto numpyCases2 = ::testing::Combine(
//     ::testing::ValuesIn(numpyShapes2),
//     ::testing::ValuesIn(inputPrecision),
//     ::testing::Values(ngraph::op::AutoBroadcastSpec::NUMPY),
//     ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
// );

//INSTANTIATE_TEST_CASE_P(smoke_TestsSelectOp_numpy2, KmbSelectLayerTest_MLIR, numpyCases2, KmbSelectLayerTest::getTestCaseName);

}  // namespace
