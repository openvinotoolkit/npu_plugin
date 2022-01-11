// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbScatterElementsUpdateLayerTest: public ScatterElementsUpdateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbScatterElementsUpdateLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

//indices should not be random value
const std::vector<size_t> idxValue {0, 2, 4, 6, 1, 3, 5, 7};

//inShape, updateShape, axis
const std::vector<axisShapeInShape> params = {
    {{10,12,15}, {1,2,4}, 0},
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterElementsUpdate,
        KmbScatterElementsUpdateLayerTest,
        testing::Combine(
           testing::ValuesIn(params),
           testing::Values(idxValue),                         // indices values
           testing::Values(InferenceEngine::Precision::FP16), // input prec
           testing::Values(InferenceEngine::Precision::I32),  // indices prec
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
       KmbScatterElementsUpdateLayerTest::getTestCaseName
);

}  // namespace
