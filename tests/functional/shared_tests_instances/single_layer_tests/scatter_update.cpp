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

//indices should not be random value
const std::vector<std::vector<int64_t>> idxValue = {
        {0, 2, 4, 6, 1, 3, 5, 7}
};

//inShape, indicesShape, updateShape, axis
const std::vector<axisUpdateShapeInShape> toyParam {
   {{10, 9, 10, 9, 10}, {8},     {8, 9, 10, 9, 10},  0}, //OK
   {{10, 16, 12, 15},   {2,4},   {2, 4, 16, 12, 15}, 0}, //OK
   {{10, 16, 12, 15},   {8},     {8, 16, 12, 15},    0}, //OK
   {{10, 16, 12},       {2,2,2}, {2, 2, 2, 16, 12},  0}, //OK
   {{10, 9, 10, 9},     {4,2},   {4, 2, 9, 10, 9},   0}, //OK
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterUpdateToy,
        KmbScatterUpdateLayerTest,
        testing::Combine(
           testing::ValuesIn(toyParam),
           testing::ValuesIn(idxValue),                       //indices values
           testing::Values(InferenceEngine::Precision::FP16), // input prec
           testing::Values(InferenceEngine::Precision::I32),  // indices prec
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbScatterUpdateLayerTest::getTestCaseName
);

}  // namespace
