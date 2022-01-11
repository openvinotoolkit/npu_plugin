// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbScatterElementsUpdateLayerTest: public ScatterElementsUpdateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto &params = std::get<0>(GetParam());
        const auto  axis   = std::get<2>(params);
        if (axis < 0) {
            throw LayerTestsUtils::KmbSkipTestException("Runtime only supports positive axis, actual=" + std::to_string(axis));
        }
    }
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
const std::vector<axisShapeInShape> params =
{
    {{10,12,15},  {1,2,4},   0},
    {{10,12,15},  {1,2,4},   1},
    {{10,12,15},  {1,2,4},   2},

    {{10,12,15},  {2,2,2},   0},
    {{10,12,15},  {2,2,2},   1},
    {{10,12,15},  {2,2,2},   2},

    {{15,9,8,12}, {1,2,2,2}, 0},
    {{15,9,8,12}, {1,2,2,2}, 1},
    {{15,9,8,12}, {1,2,2,2}, 2},
    {{15,9,8,12}, {1,2,2,2}, 3},

    {{15,9,8,12}, {1,2,1,4}, 0},
    {{15,9,8,12}, {1,2,1,4}, 1},
    {{15,9,8,12}, {1,2,1,4}, 2},
    {{15,9,8,12}, {1,2,1,4}, 3},
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
