// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reorg_yolo.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbReorgYoloLayerTest : public ReorgYoloLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon
{
    void SkipBeforeLoad() override {
        auto inputShape = std::get<0>(GetParam());
        if (inputShape[0] != 1) {
            throw LayerTestsUtils::KmbSkipTestException("Runtime only supports N=1 shape");
        }
    }
};

TEST_P(KmbReorgYoloLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ngraph::Shape> inputShapesA = {
    ngraph::Shape{1, 64, 26, 26}, // openvino eg
    ngraph::Shape{1,  4,  4,  4},
    ngraph::Shape{1,  8,  4,  4},
    ngraph::Shape{2,  8,  4,  4},

  //ngraph::Shape{1, 62, 14, 14}, // fails
  //ngraph::Shape{1, 62, 34, 24},
  //ngraph::Shape{1, 24, 34, 62},
  //ngraph::Shape{1, 64,500,500},
  //ngraph::Shape{1, 26, 64, 26},
};

const std::vector<ngraph::Shape> inputShapesB = {
    ngraph::Shape{1, 9, 3, 3},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(
        smoke_ReorgYolo_a,
        KmbReorgYoloLayerTest,
        testing::Combine(
            testing::ValuesIn(inputShapesA),
            testing::Values(2), // stride
            testing::ValuesIn(netPrecisions),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
        ),
        KmbReorgYoloLayerTest::getTestCaseName
    );

INSTANTIATE_TEST_CASE_P(
        smoke_ReorgYolo_b,
        KmbReorgYoloLayerTest,
        testing::Combine(
            testing::ValuesIn(inputShapesB),
            testing::Values(3), // stride
            testing::ValuesIn(netPrecisions),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
        ),
        KmbReorgYoloLayerTest::getTestCaseName
    );

}  // namespace
