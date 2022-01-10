// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbScatterUpdateLayerTest: public ScatterUpdateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto &params = std::get<0>(GetParam());
        const auto  axis   = std::get<3>(params);
        if (axis != 0) {
            throw LayerTestsUtils::KmbSkipTestException("Runtime only supports axis=0 config, actual=" + std::to_string(axis));
        }
    }
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
const std::vector<axisUpdateShapeInShape> params {
   {{10, 9, 10, 9, 10}, {8},     {8, 9, 10, 9, 10},  0},
   {{10, 16, 12, 15},   {2,4},   {2, 4, 16, 12, 15}, 0},
   {{10, 16, 12, 15},   {8},     {8, 16, 12, 15},    0},
   {{10, 16, 12},       {2,2,2}, {2, 2, 2, 16, 12},  0},
   {{10, 9, 10, 9},     {4,2},   {4, 2, 9, 10, 9},   0},
   {{10, 16, 12},       {1,8},   {1, 8, 16, 12},     0},
   {{8, 9, 10, 3},      {4,2},   {4, 2, 9, 10, 3},   0},
   {{11, 4, 3, 4},      {2,4},   {2, 4, 4, 3, 4},    0},
   {{12, 9, 11, 2},     {4,2},   {4, 2, 9, 11, 2},   0}
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterUpdateToy,
        KmbScatterUpdateLayerTest,
        testing::Combine(
           testing::ValuesIn(params),
           testing::ValuesIn(idxValue),                       // indices values
           testing::Values(InferenceEngine::Precision::FP16), // input prec
           testing::Values(InferenceEngine::Precision::I32),  // indices prec
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbScatterUpdateLayerTest::getTestCaseName
);

}  // namespace
