// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "kmb_test_report.hpp"

namespace LayerTestsDefinitions {

class KmbInterpolateLayerTest: public InterpolateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbInterpolateLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbInterpolateLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbInterpolateLayerTest_VPU4000: public InterpolateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void ConfigureNetwork() override {
        // default f32 network precision attempts to insert IE.Convert ops to/from u8/f16, but these are not available yet
        if(cnnNetwork.getInputsInfo().begin()->second->getPrecision() == InferenceEngine::Precision::U8){
          cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);
        }
        else if(cnnNetwork.getInputsInfo().begin()->second->getPrecision() == InferenceEngine::Precision::FP32){
          cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
          cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }

    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException("Missing M2I runtime support");
    }
};

TEST_P(KmbInterpolateLayerTest_VPU4000, CompareWithRefs_MLIR_VPU4000) {
    useCompilerMLIR();
    setPlatformVPU4000();
    setDefaultHardwareModeMLIR();
    Run();
}

class KmbInterpolate1Test: public Interpolate1LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

public:
// Reference version for Interpolate-1 layer doesn't implemented in OpenVino.
// It is enough to check the conversion of Interpolate-1 layer to Interpolate-4 layer,
// as test for Interpolate-4 has already exist. So skip Validate step.
    void Validate() override {
        std::cout << "Skip Validate() for Interpolate1" << std::endl;
    }
};

TEST_P(KmbInterpolate1Test, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbInterpolateLayerTest_VPU3720 : public InterpolateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbInterpolateLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inShapes = {
    {1, 10, 30, 30},
};

const std::vector<std::vector<size_t>> targetShapes = {
    {40, 40},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear,
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearest = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearest2x = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearestMore = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesWithoutNearest = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::simple,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v4::Interpolate::NearestMode::floor,
        ngraph::op::v4::Interpolate::NearestMode::ceil,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestModeMore = {
        ngraph::op::v4::Interpolate::NearestMode::floor,
};

const std::vector<std::vector<size_t>> pads = {
    // {0, 0, 1, 1},
    {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
    false,
};

const std::vector<double> cubeCoefs = {
    -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    {2, 3}
};

const std::vector<std::vector<float>> defaultScales = {
    {1.33333f, 1.33333f}
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
        // ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
};

std::map<std::string, std::string> additional_config = {};

auto interpolateCasesNearestMode = [] (auto scales) {
    return ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesNearest),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(scales));
};

auto interpolateCasesWithoutNearestMode = [] (auto scales) {
    return ::testing::Combine(
            ::testing::ValuesIn(modesWithoutNearest), ::testing::ValuesIn(shapeCalculationMode),
            ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
            ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
            ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(defaultAxes),
            ::testing::ValuesIn(scales));
};

const auto interpolateM2IMode = ::testing::Combine(
        ::testing::Values(nearestMode[0]),
        ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::sizes),
        ::testing::Values(coordinateTransformModesWithoutNearest[0]),
        ::testing::Values(defaultNearestMode[0]),
        ::testing::Values(false),                           //antialias
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}), //pads_begin
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}), //pads_end
        ::testing::Values(0.0),                             //cube_coeff
        ::testing::Values(std::vector<int64_t>{2, 3}),      //axes
        ::testing::Values(std::vector<float>{0.0f, 0.0f})); //scales

const auto interpCminorM2IMode = ::testing::Combine(
        ::testing::Values(nearestMode[0]),
        ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::sizes),
        ::testing::Values(coordinateTransformModesWithoutNearest[0]),
        ::testing::Values(defaultNearestMode[0]),
        ::testing::Values(false),                           //antialias
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}), //pads_begin
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}), //pads_end
        ::testing::Values(0.0),                             //cube_coeff
        ::testing::Values(std::vector<int64_t>{1, 2}),      //axes
        ::testing::Values(std::vector<float>{0.0f, 0.0f})); //scales

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_mode, KmbInterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearestMode(defaultScales),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config)),
                            KmbInterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_without_nearest, KmbInterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestMode(defaultScales),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config)),
                            KmbInterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_nearest_mode_VPU3720, KmbInterpolateLayerTest_VPU3720, ::testing::Combine(
        interpolateCasesNearestMode(defaultScales),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config)),
                            KmbInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_without_nearest_VPU3720, KmbInterpolateLayerTest_VPU3720, ::testing::Combine(
        interpolateCasesWithoutNearestMode(defaultScales),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config)),
                            KmbInterpolateLayerTest_VPU3720::getTestCaseName);

const std::vector<std::vector<size_t>> inShapesForTiling = {
        {1, 32, 32, 64},
};

const std::vector<std::vector<size_t>> targetShapesForTiling = {
           { 32,  64}, // x1.00
           {128, 256}, // x4.00
        // {136, 272}, // x4.25
        // {144, 288}, // x4.50
        // {152, 304}, // x4.75
};

auto makeScales = [] (float uniformScale) {
        const std::vector<std::vector<float>> scales = {{uniformScale, uniformScale}};
        return scales;
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_with_tiling_VPU3720, KmbInterpolateLayerTest_VPU3720, ::testing::Combine(
        interpolateCasesNearestMode(makeScales(1.f)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapesForTiling),
        ::testing::ValuesIn(targetShapesForTiling),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config)),
                         KmbInterpolateLayerTest_VPU3720::getTestCaseName);

const std::vector<std::string> mode = {"nearest", "linear"};
const std::vector<ngraph::AxisSet> axes ={{2, 3}};

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_1, KmbInterpolate1Test, ::testing::Combine(
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::ValuesIn(mode),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbInterpolate1Test::getTestCaseName);

//VPU4000
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_M2I_u8,
       KmbInterpolateLayerTest_VPU4000,
       ::testing::Combine(
          interpolateM2IMode,
          ::testing::Values(InferenceEngine::Precision::U8),
          ::testing::Values(InferenceEngine::Precision::U8),
          ::testing::Values(InferenceEngine::Precision::U8),
          ::testing::Values(InferenceEngine::Layout::NCHW),
          ::testing::Values(InferenceEngine::Layout::NCHW),
          ::testing::Values(std::vector<size_t>{1, 3, 256, 256}), //in-Shape
          ::testing::Values(std::vector<size_t>{224, 224}),       //out-Shape for given axes
          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
          ::testing::Values(additional_config)
       ),
       KmbInterpolateLayerTest_VPU4000::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_M2I_fp16,
       KmbInterpolateLayerTest_VPU4000,
       ::testing::Combine(
          interpolateM2IMode,
          ::testing::Values(InferenceEngine::Precision::FP16),
          ::testing::Values(InferenceEngine::Precision::FP16),
          ::testing::Values(InferenceEngine::Precision::FP16),
          ::testing::Values(InferenceEngine::Layout::NCHW),
          ::testing::Values(InferenceEngine::Layout::NCHW),
          ::testing::Values(std::vector<size_t>{1, 3, 256, 256}),
          ::testing::Values(std::vector<size_t>{224, 224}),
          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
          ::testing::Values(additional_config)
       ),
       KmbInterpolateLayerTest_VPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_M2I_u8_Cminor,
       KmbInterpolateLayerTest_VPU4000,
       ::testing::Combine(
          interpCminorM2IMode,
          ::testing::Values(InferenceEngine::Precision::U8),
          ::testing::Values(InferenceEngine::Precision::U8),
          ::testing::Values(InferenceEngine::Precision::U8),
          ::testing::Values(InferenceEngine::Layout::NCHW),
          ::testing::Values(InferenceEngine::Layout::NCHW),
          ::testing::Values(std::vector<size_t>{1, 256, 256, 3}), //in-Shape
          ::testing::Values(std::vector<size_t>{224, 224}),       //out-Shape for given axes
          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
          ::testing::Values(additional_config)
       ),
       KmbInterpolateLayerTest_VPU4000::getTestCaseName);

} // namespace
