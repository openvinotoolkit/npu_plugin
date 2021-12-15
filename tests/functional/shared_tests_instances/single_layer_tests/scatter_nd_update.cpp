// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_ND_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbScatterNDUpdateLayerTest: public ScatterNDUpdateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbScatterNDUpdateLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// inShape, indiceShape, indicesValues, updateShape
const sliceSelectInShape scatterNDUpdParams[] = {
  { {8}, {4,1}, {4,3,1,7}, {4} },     //openvino eg.1
  { {4,4,4}, {2,1}, {0,2}, {2,4,4} }, //openvino eg.2
  { {1}, {1,1}, {0}, {1} },
  { {2,2}, {2,1}, {1,0}, {2,2} },
  { {2,2}, {2,2}, {0,0, 1,1}, {2} },
  { {3,3,3}, {2,1}, {0,2}, {2,3,3} },
  { {3,3,3}, {2,2,3}, {0,0,0, 2,2,2, 1,0,0, 1,2,2}, {2,2} },
  { {3,3,3}, {2,2}, {0,0, 2,2}, {2,3} },
  { {3,3,3}, {2,3}, {0,0,0, 2,2,2}, {2} },
  { {4,4,4}, {2,2,2}, {0,0, 2,2, 1,1, 3,3}, {2,2,4} }
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterNDUpdate,
        KmbScatterNDUpdateLayerTest,
        testing::Combine(
           testing::ValuesIn(scatterNDUpdParams),
           testing::Values(InferenceEngine::Precision::FP16), // network
           testing::Values(InferenceEngine::Precision::I32),  // indices
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbScatterNDUpdateLayerTest::getTestCaseName
);

}  // namespace
