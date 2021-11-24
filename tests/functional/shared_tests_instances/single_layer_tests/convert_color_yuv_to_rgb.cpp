// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convert_color_nv12.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvertColorNV12LayerTest: public ConvertColorNV12LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    // void ConfigureNetwork() override {
    //    cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    //   cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    // }
};

TEST_P(KmbConvertColorNV12LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// N,H,W,C
ov::Shape inShapes[] = {
 {1,   4,   8, 1},
 {1,  64,  32, 1},
 {3, 128, 128, 1}
};

ov::element::Type dType = ov::element::f16;

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorNV12,
        KmbConvertColorNV12LayerTest,
        testing::Combine(
           testing::ValuesIn(inShapes),
           testing::Values(dType),   // elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true),    // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorNV12LayerTest::getTestCaseName
);

}  // namespace
