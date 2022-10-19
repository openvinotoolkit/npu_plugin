//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convert_color_nv12.hpp"
#include "single_layer_tests/convert_color_i420.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvertColorNV12LayerTest: public ConvertColorNV12LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorNV12LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbConvertColorI420LayerTest: public ConvertColorI420LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorI420LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbConvertColorNV12LayerTest_VPU4000: public ConvertColorNV12LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
     // default out network precision seems fp32, which 'inserts' an undesirable trailing Convert(u8->f32)
     void ConfigureNetwork() override {
        if(cnnNetwork.getInputsInfo().begin()->second->getPrecision() == InferenceEngine::Precision::U8){
          cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);
        }
     }

     void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Missing M2I runtime support");
     }
};
class KmbConvertColorI420LayerTest_VPU4000: public ConvertColorI420LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
     void ConfigureNetwork() override {
        if(cnnNetwork.getInputsInfo().begin()->second->getPrecision() == InferenceEngine::Precision::U8){
          cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);
        }
     }

     // Temp workaround for OpenVino issue #11736
     // (not instantiating proper op for Single-plane I420 (generates NV12 instead)
     void SetUp() override {
         ov::Shape inputShape;
         ov::element::Type ngPrc;
         bool conversionToRGB, singlePlane;
         abs_threshold = 1.0f; // I420 conversion can use various algorithms, thus some absolute deviation is allowed
         threshold = 1.f; // Ignore relative comparison for I420 convert (allow 100% relative deviation)
         std::tie(inputShape, ngPrc, conversionToRGB, singlePlane, targetDevice) = GetParam();

         if (singlePlane) {
             inputShape[1] = inputShape[1] * 3 / 2;
             auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
             std::shared_ptr<ov::Node> convert_color;
             if (conversionToRGB) {
                convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
             } else {
                convert_color = std::make_shared<ov::op::v8::I420toBGR>(param);
             }
         function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param}, "ConvertColorI420");
        }else{
            throw LayerTestsUtils::KmbSkipTestException("Multi-plane not supported");
        }
    }

    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Missing M2I runtime support");
    }
};

TEST_P(KmbConvertColorNV12LayerTest_VPU4000, CompareWithRefs_MLIR_VPU4000) {
    useCompilerMLIR();
    setPlatformVPU4000();
    setDefaultHardwareModeMLIR();
    Run();
}
TEST_P(KmbConvertColorI420LayerTest_VPU4000, CompareWithRefs_MLIR_VPU4000) {
    useCompilerMLIR();
    setPlatformVPU4000();
    setDefaultHardwareModeMLIR();
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

ov::element::Type dTypes[] = {
  ov::element::f16,
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorNV12,
        KmbConvertColorNV12LayerTest,
        testing::Combine(
           testing::ValuesIn(inShapes),
           testing::ValuesIn(dTypes),    // elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true, false), // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorNV12LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorI420,
        KmbConvertColorI420LayerTest,
        testing::Combine(
           testing::ValuesIn(inShapes),
           testing::ValuesIn(dTypes),    // elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true, false), // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorNV12LayerTest::getTestCaseName
);

//VPU4000
INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorNV12_M2I,
        KmbConvertColorNV12LayerTest_VPU4000,
        testing::Combine(
           testing::Values(ov::Shape{1, 240, 320, 1}), // QVGA
           testing::Values(ov::element::u8),// elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true), // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorNV12LayerTest_VPU4000::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorI420_M2I,
        KmbConvertColorI420LayerTest_VPU4000,
        testing::Combine(
           testing::Values(ov::Shape{1, 240, 320, 1}), // QVGA
           testing::Values(ov::element::u8),// elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true), // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorI420LayerTest_VPU4000::getTestCaseName
);

}  // namespace
