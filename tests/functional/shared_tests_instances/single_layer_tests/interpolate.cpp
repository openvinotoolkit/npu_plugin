// Copyright (C) 2019-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "kmb_test_report.hpp"
#include "single_layer_tests/interpolate.hpp"
#include "vpux_private_config.hpp"

namespace LayerTestsDefinitions {

class VPUXInterpolateLayerTest : public InterpolateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXInterpolateLayerTest_VPU3700 : public VPUXInterpolateLayerTest {};
class VPUXInterpolateLayerTest_VPU3720 : public VPUXInterpolateLayerTest {};

class VPUXInterpolateLayerSETest_VPU3720 : public VPUXInterpolateLayerTest {
    void ConfigureNetwork() override {
        configuration[VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS)] = "enable-se-ptrs-operations=true";
    }
};

TEST_P(VPUXInterpolateLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXInterpolateLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXInterpolateLayerSETest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

class VPUXInterpolate1LayerTest_VPU3700 :
        public Interpolate1LayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
public:
    // Reference version for Interpolate-1 layer doesn't implemented in OpenVino.
    // It is enough to check the conversion of Interpolate-1 layer to Interpolate-4 layer,
    // as test for Interpolate-4 has already exist. So skip Validate step.
    void Validate() override {
        std::cout << "Skip Validate() for Interpolate1" << std::endl;
    }
};

TEST_P(VPUXInterpolate1LayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 10, 30, 30},
};

const std::vector<std::vector<size_t>> targetShapes = {
        {40, 40},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
        ngraph::op::v4::Interpolate::InterpolateMode::CUBIC,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> linearModes = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> linearOnnxMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearest = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearest2x = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModeAsymmetric = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesWithoutNearest = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::CEIL,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestModeFloor = {
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
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

const std::vector<std::vector<int64_t>> defaultAxes = {{2, 3}};

const std::vector<std::vector<float>> defaultScales = {{1.33333f, 1.33333f}};

const std::vector<std::vector<int64_t>> allAxes = {{0, 1, 2, 3}};
const std::vector<std::vector<float>> allScales = {{1.f, 1.f, 1.33333f, 1.33333f}};
const std::vector<std::vector<size_t>> allScalescTargetShapes = {
        {1, 10, 40, 40},
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        // ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
};

const std::vector<InferenceEngine::Layout> layout = {InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC};

std::map<std::string, std::string> additional_config = {};

auto interpolateCasesNearestMode = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(defaultAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesLinearOnnxMode = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(linearOnnxMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(defaultAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesWithoutNearestMode = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(modesWithoutNearest), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(defaultAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesAllAxes = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(allAxes), ::testing::ValuesIn(scales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_mode, VPUXInterpolateLayerTest_VPU3700,
                         ::testing::Combine(interpolateCasesNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_without_nearest, VPUXInterpolateLayerTest_VPU3700,
                         ::testing::Combine(interpolateCasesWithoutNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_nearest_mode_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_without_nearest_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::ValuesIn(layout),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_all_axes_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesAllAxes(allScales), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(allScalescTargetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

const std::vector<std::vector<size_t>> inShapesForTiling = {
        {1, 32, 32, 64},
};

const std::vector<std::vector<size_t>> targetShapesForTiling = {
        {32, 64},    // x1.00
        {128, 256},  // x4.00
                     // {136, 272}, // x4.25
                     // {144, 288}, // x4.50
                     // {152, 304}, // x4.75
};

auto makeScales = [](float uniformScale) {
    const std::vector<std::vector<float>> scales = {{uniformScale, uniformScale}};
    return scales;
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_with_tiling_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesNearestMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapesForTiling),
                                            ::testing::ValuesIn(targetShapesForTiling),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_with_align_corners_tiling_VPU3720, VPUXInterpolateLayerTest_VPU3720,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::ANY),
                           ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(inShapesForTiling),
                           ::testing::ValuesIn(targetShapesForTiling),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                           ::testing::Values(additional_config)),
        VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_with_align_corners_tiling_reduce_size_VPU3720,
                         VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 257, 257}}),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{17, 17}}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

// test different channels
const std::vector<std::vector<size_t>> inShapesForNHWCLayoutOptimize = {
        /*{1, 1, 32, 32},*/ {1, 2, 32, 32},
        {1, 3, 32, 32},
        {1, 4, 32, 32},
        {1, 5, 32, 32},
        {1, 6, 32, 32},
        {1, 7, 32, 32},
        {1, 8, 32, 32},
};

const std::vector<std::vector<size_t>> outShapesForNHWCLayoutOptimizeSmokePrecommit = {
        {64, 64},
};

// test different output shapes
const std::vector<std::vector<size_t>> outShapesForNHWCLayoutOptimizeSmoke = {
        {16, 16}, {24, 24}, {32, 32}, {48, 48}, {64, 64},
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_NHWCLayout_optimize_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesForNHWCLayoutOptimize),
                                            ::testing::ValuesIn(outShapesForNHWCLayoutOptimizeSmokePrecommit),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NHWCLayout_optimize_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesForNHWCLayoutOptimize),
                                            ::testing::ValuesIn(outShapesForNHWCLayoutOptimizeSmoke),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

const std::vector<std::string> mode = {"nearest", "linear"};
const std::vector<ngraph::AxisSet> axes = {{2, 3}};

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_1, VPUXInterpolate1LayerTest_VPU3700,
                        ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Layout::NCHW),
                                           ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                           ::testing::ValuesIn(mode), ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXInterpolate1LayerTest_VPU3700::getTestCaseName);

const std::vector<std::vector<size_t>> inShapesLargeNHWC = {
        {1, 224, 224, 3},
};
const std::vector<std::vector<size_t>> inShapesLargeNCHW = {
        {1, 3, 224, 224},
};

const std::vector<std::vector<size_t>> targetShapesLarge = {
        {300, 300},
};

const std::vector<std::vector<int64_t>> nhwcAxes = {{1, 2}};
const std::vector<std::vector<int64_t>> nchwAxes = {{2, 3}};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> interpolateMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR, ngraph::op::v4::Interpolate::InterpolateMode::CUBIC};

auto interpolateCasesWithoutNearestModeLargerNHWC = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(interpolateMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(scales));
};
auto interpolateCasesWithoutNearestModeLargerNCHW = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(interpolateMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

// test case for fixing input NCHW layout axes=2,3 incorrect result issue
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NCHWinput_NCHWlayout_NCHWaxes_VPU3720, VPUXInterpolateLayerTest_VPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::NCHW),
                           ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::ValuesIn(inShapesLargeNCHW),
                           ::testing::ValuesIn(targetShapesLarge),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                           ::testing::Values(additional_config)),
        VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

// test case for input NCHW layout axes=1,2 support
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NHWCinput_NCHWlayout_NHWCaxes_VPU3720, VPUXInterpolateLayerTest_VPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::NCHW),
                           ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::ValuesIn(inShapesLargeNHWC),
                           ::testing::ValuesIn(targetShapesLarge),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                           ::testing::Values(additional_config)),
        VPUXInterpolateLayerTest_VPU3720::getTestCaseName);
// test case for input NHWC layout axes=1,2 support
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NHWCinput_NHWClayout_NHWCaxes_VPU3720, VPUXInterpolateLayerTest_VPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Precision::FP16),
                           ::testing::Values(InferenceEngine::Layout::NHWC),
                           ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::ValuesIn(inShapesLargeNHWC),
                           ::testing::ValuesIn(targetShapesLarge),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                           ::testing::Values(additional_config)),
        VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

// test case for 2D or 3D input
const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationModeSizeScale = {
        // ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
};

const std::vector<std::vector<size_t>> pads3D = {
        {0, 0, 0},
};

const std::vector<std::vector<size_t>> inShapes3D = {
        {8, 64, 2},
};

const std::vector<std::vector<float>> scales3D = {
        {1.0f, 1.0f, 2.0f},
};

const std::vector<std::vector<size_t>> targetShapes3D = {
        {8, 64, 4},
};

const std::vector<std::vector<int64_t>> AxesInput3D = {
        {0, 1, 2},
};

auto interpolateCaseNearestModeNC_Nearst_Input3D = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads3D), ::testing::ValuesIn(pads3D), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(AxesInput3D), ::testing::ValuesIn(scales3D));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input3D_VPU3720,
                         VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCaseNearestModeNC_Nearst_Input3D(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::CHW),
                                            ::testing::Values(InferenceEngine::Layout::CHW),
                                            ::testing::ValuesIn(inShapes3D), ::testing::ValuesIn(targetShapes3D),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

const std::vector<std::vector<size_t>> pads2D = {
        {0, 0},
};

const std::vector<std::vector<size_t>> inShapes2D = {
        {64, 2},
};

const std::vector<std::vector<float>> scales2D = {
        {1.0f, 2.0f},
};

const std::vector<std::vector<size_t>> targetShapes2D = {
        {64, 4},
};

const std::vector<std::vector<int64_t>> AxesInput2D = {
        {0, 1},
};

auto interpolateCaseNearestModeNC_Nearst_Input2D = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads2D), ::testing::ValuesIn(pads2D), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(AxesInput2D), ::testing::ValuesIn(scales2D));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input2D_VPU3720,
                         VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCaseNearestModeNC_Nearst_Input2D(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NC),
                                            ::testing::Values(InferenceEngine::Layout::NC),
                                            ::testing::ValuesIn(inShapes2D), ::testing::ValuesIn(targetShapes2D),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

// NEAREST cases | Axes=1,2 | Layout: NCHW and NHWC
auto interpolateCasesNearestModeAxes12 = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest), ::testing::ValuesIn(nearestModes),
                              ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
                              ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes),
                              ::testing::ValuesIn(defaultScales));
};
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCHWlayout_NHWCaxes_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(inShapesLargeNHWC),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NHWClayout_NHWCaxes_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesLargeNHWC),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modePytorchHalfPixel = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
};
const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModePytorchHalfPixel = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
};
auto interpolateParamsPytorchHalfPixel = []() {
    return ::testing::Combine(::testing::ValuesIn(modePytorchHalfPixel), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModePytorchHalfPixel),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));
};
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_PytorchHalfPixel_Tiling_Upscale_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateParamsPytorchHalfPixel(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(std::vector<size_t>({1, 32, 68, 120})),
                                            ::testing::Values(std::vector<size_t>({136, 240})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_PytorchHalfPixel_Tiling_Downscale_VPU3720, VPUXInterpolateLayerTest_VPU3720,
                         ::testing::Combine(interpolateParamsPytorchHalfPixel(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(std::vector<size_t>({1, 3, 270, 480})),
                                            ::testing::Values(std::vector<size_t>({135, 240})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerTest_VPU3720::getTestCaseName);

//
// SE Interpolate
//

const std::vector<std::vector<float>> seInterpolateScales = {{2.0f, 2.0f}, {5.0f, 5.0f}};

const std::vector<std::vector<size_t>> seInterpolateInputShapes = {
        {1, 48, 15, 15},
};

const std::vector<std::vector<size_t>> seInterpolateTargetShapes = {
        {},
};

auto seInterpolateParamsNearest = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric), ::testing::ValuesIn(nearestModes),
                              ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
                              ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes),
                              ::testing::ValuesIn(seInterpolateScales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, VPUXInterpolateLayerSETest_VPU3720,
                         ::testing::Combine(seInterpolateParamsNearest(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerSETest_VPU3720::getTestCaseName);

auto seInterpolateParamsLinear = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear, VPUXInterpolateLayerSETest_VPU3720,
                         ::testing::Combine(seInterpolateParamsLinear(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXInterpolateLayerSETest_VPU3720::getTestCaseName);

}  // namespace
