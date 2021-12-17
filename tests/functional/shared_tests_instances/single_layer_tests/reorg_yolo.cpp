// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reorg_yolo.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbReorgYoloLayerTest : public ReorgYoloLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbReorgYoloLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ngraph::Shape> inputShapes = {
    ngraph::Shape{1, 64, 26, 26} // openvino eg
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(
        smoke_ReorgYolo,
        KmbReorgYoloLayerTest,
        testing::Combine(
            testing::ValuesIn(inputShapes),
            testing::Values(2),              // stride
            testing::ValuesIn(netPrecisions),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
        ),
        KmbReorgYoloLayerTest::getTestCaseName
    );

}  // namespace
