// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbScatterUpdateLayerTest: public ScatterUpdateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbScatterUpdateLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {


// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 16, 12, 15}, {{{2, 2, 2}, {0, 1, 2, 3}}, {{2, 4}, {0, 1, 2, 3}}, {{8}, {0, 1, 2, 3}}}},
    {{10, 9, 10, 9, 10}, {{{8}, {0, 1, 2, 3, 4}}, {{4, 2}, {0, 1, 2, 3, 4}}}},
    {{10, 9, 10, 9, 10, 12}, {{{8}, {0, 1, 2, 3, 4, 5}}}},
    {{10, 16, 12, 15}, {{{2, 4}, {0, 1, 2, 3}}, {{8}, {-1, -2, -3, -4}}}},
    {{10, 9, 10, 9, 10}, {{{8}, {-3, -1, 0, 2, 4}}, {{4, 2}, {-2, 2}}}},
};
//indices should not be random value
const std::vector<std::vector<int64_t>> idxValue = {
        {0, 2, 4, 6, 1, 3, 5, 7}
};

/*
INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterUpdate,
        KmbScatterUpdateLayerTest,
        testing::Combine(
           testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)),
           testing::ValuesIn(idxValue),
           testing::Values(InferenceEngine::Precision::FP16), // input prec
           testing::Values(InferenceEngine::Precision::I32),  // indices prec
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbScatterUpdateLayerTest::getTestCaseName
);
*/
//==============================================================================
const axisUpdateShapeInShape toyParam {
    {10, 9, 10, 9, 10}, //in shape
    { 8},               //indices shape
    { 8, 9, 10, 9, 10}, //update shape
     0                  //axis
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterUpdateToy,
        KmbScatterUpdateLayerTest,
        testing::Combine(
           testing::Values(toyParam),
           testing::Values(std::vector<int64_t>{0, 2, 4, 6, 1, 3, 5, 7}), //indices values
           testing::Values(InferenceEngine::Precision::FP16), // input prec
           testing::Values(InferenceEngine::Precision::I32),  // indices prec
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbScatterUpdateLayerTest::getTestCaseName
);

}  // namespace
