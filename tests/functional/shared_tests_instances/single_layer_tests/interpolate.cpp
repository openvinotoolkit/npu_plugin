// Copyright (C) 2019-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_properties.hpp"

namespace LayerTestsDefinitions {

class InterpolateLayerTestCommon :
        public InterpolateLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class InterpolateLayerTest_NPU3700 : public InterpolateLayerTestCommon {};
class InterpolateLayerTest_NPU3720 : public InterpolateLayerTestCommon {};

class InterpolateLayerSETest_NPU3720 : public InterpolateLayerTestCommon {
    void ConfigureNetwork() override {
        configuration[ov::intel_vpux::compilation_mode_params.name()] = "enable-se-ptrs-operations=true";
    }
};

using InterpolateLayerSETest_NPU3720_ELF = InterpolateLayerSETest_NPU3720;

TEST_P(InterpolateLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(InterpolateLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(InterpolateLayerSETest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(InterpolateLayerSETest_NPU3720_ELF, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    useELFCompilerBackend();
    Run();
}

class Interpolate1LayerTest_NPU3700 :
        public Interpolate1LayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
public:
    // Reference version for Interpolate-1 layer doesn't implemented in OpenVino.
    // It is enough to check the conversion of Interpolate-1 layer to Interpolate-4 layer,
    // as test for Interpolate-4 has already exist. So skip Validate step.
    void Validate() override {
        std::cout << "Skip Validate() for Interpolate1" << std::endl;
    }
};

TEST_P(Interpolate1LayerTest_NPU3700, HW) {
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

const std::vector<std::vector<int64_t>> nhwcAxes = {{1, 2}};
const std::vector<std::vector<int64_t>> nchwAxes = {{2, 3}};

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
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesLinearOnnxMode = [](auto scales) {
    return ::testing::Combine(::testing::Values(linearModes[1]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesWithoutNearestMode = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(modesWithoutNearest), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesAllAxes = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(allAxes), ::testing::ValuesIn(scales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_mode, InterpolateLayerTest_NPU3700,
                         ::testing::Combine(interpolateCasesNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_without_nearest, InterpolateLayerTest_NPU3700,
                         ::testing::Combine(interpolateCasesWithoutNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_nearest_mode, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_without_nearest, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestMode(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::ValuesIn(layout),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_all_axes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesAllAxes(allScales), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapes), ::testing::ValuesIn(allScalescTargetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_with_tiling, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesNearestMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapesForTiling),
                                            ::testing::ValuesIn(targetShapesForTiling),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_with_tiling_2x, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesNearestMode(makeScales(2.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 3, 160, 160})),
                                            ::testing::Values(std::vector<size_t>({320, 320})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_with_align_corners_tiling, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inShapesForTiling),
                                            ::testing::ValuesIn(targetShapesForTiling),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_with_align_corners_tiling_reduce_size,
                         InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 257, 257}}),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{17, 17}}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_with_align_corners_2x, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 512, 7, 7}}),
                                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{14, 14}}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_NHWCLayout_optimize, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesForNHWCLayoutOptimize),
                                            ::testing::ValuesIn(outShapesForNHWCLayoutOptimizeSmokePrecommit),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NHWCLayout_optimize, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesForNHWCLayoutOptimize),
                                            ::testing::ValuesIn(outShapesForNHWCLayoutOptimizeSmoke),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

const std::vector<std::string> mode = {"nearest", "linear"};
const std::vector<ngraph::AxisSet> axes = {{2, 3}};

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_1, Interpolate1LayerTest_NPU3700,
                        ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Precision::FP16),
                                           ::testing::Values(InferenceEngine::Layout::NCHW),
                                           ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                           ::testing::ValuesIn(mode), ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        Interpolate1LayerTest_NPU3700::getTestCaseName);

const std::vector<std::vector<size_t>> inShapesLargeNHWC = {
        {1, 112, 112, 3},
};
const std::vector<std::vector<size_t>> inShapesLargeNCHW = {
        {1, 3, 112, 112},
};

const std::vector<std::vector<size_t>> targetShapesLarge = {
        {150, 150},
};

const std::vector<std::vector<size_t>> linearInShapesLargeNHWC = {
        {1, 80, 80, 3},
};
const std::vector<std::vector<size_t>> linearInShapesLargeNCHW = {
        {1, 3, 80, 80},
};

const std::vector<std::vector<size_t>> linearTargetShapesLarge = {
        {120, 120},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> interpolateMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::CUBIC};

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
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_without_nearest_NCHWinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestModeLargerNCHW(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(inShapesLargeNCHW),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

// test case for input NCHW layout axes=1,2 support
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_without_nearest_NHWCinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(inShapesLargeNHWC),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

// test case for input NHWC layout axes=1,2 support
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_without_nearest_NHWCinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesLargeNHWC),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input3D, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCaseNearestModeNC_Nearst_Input3D(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::CHW),
                                            ::testing::Values(InferenceEngine::Layout::CHW),
                                            ::testing::ValuesIn(inShapes3D), ::testing::ValuesIn(targetShapes3D),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input2D, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCaseNearestModeNC_Nearst_Input2D(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NC),
                                            ::testing::Values(InferenceEngine::Layout::NC),
                                            ::testing::ValuesIn(inShapes2D), ::testing::ValuesIn(targetShapes2D),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

// NEAREST cases | Axes=1,2 | Layout: NCHW and NHWC
auto interpolateCasesNearestModeAxes12 = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest), ::testing::ValuesIn(nearestModes),
                              ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
                              ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes),
                              ::testing::ValuesIn(defaultScales));
};
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(inShapesLargeNHWC),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(inShapesLargeNHWC),
                                            ::testing::ValuesIn(targetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

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
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_PytorchHalfPixel_Tiling_Upscale, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateParamsPytorchHalfPixel(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(std::vector<size_t>({1, 32, 68, 120})),
                                            ::testing::Values(std::vector<size_t>({136, 240})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_PytorchHalfPixel_Tiling_Downscale, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateParamsPytorchHalfPixel(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(std::vector<size_t>({1, 3, 270, 480})),
                                            ::testing::Values(std::vector<size_t>({135, 240})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

//
// SE Interpolate
//

const std::vector<std::vector<size_t>> seInterpolateInputShapes = {
        {1, 48, 15, 15},
};

const std::vector<std::vector<float>> seInterpolateScalesForScalesCalcMode = {
        {9.0f, 10.0f},
};

const std::vector<std::vector<float>> seInterpolateScalesForSizesCalcMode = {
        {},
};

const std::vector<std::vector<size_t>> seInterpolateTargetShapesForScalesCalcMode = {
        {},
};

const std::vector<std::vector<size_t>> seInterpolateTargetShapesForSizesCalcMode = {
        {127, 141},  // (127 - 1) / (15 - 1) = 9; (141 - 1) / (15 - 1) = 10
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearestSE = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN};

auto seInterpolateParamsNearest = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModesNearestSE), ::testing::ValuesIn(nearestModes),
                              ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
                              ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes),
                              ::testing::ValuesIn(seInterpolateScalesForScalesCalcMode));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, InterpolateLayerSETest_NPU3720,
                         ::testing::Combine(seInterpolateParamsNearest(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerSETest_NPU3720::getTestCaseName);

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesLinearSE = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformAlignCorners = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

auto seInterpolateParamsLinear = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes),
                              ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES),
                              ::testing::ValuesIn(coordinateTransformModesLinearSE),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesForScalesCalcMode));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear, InterpolateLayerSETest_NPU3720,
                         ::testing::Combine(seInterpolateParamsLinear(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerSETest_NPU3720::getTestCaseName);

auto seInterpolateParamsLinearWithAlignCorners = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes),
                              ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES),
                              ::testing::ValuesIn(coordinateTransformAlignCorners),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesForSizesCalcMode));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear_Align_Corners, InterpolateLayerSETest_NPU3720,
                         ::testing::Combine(seInterpolateParamsLinearWithAlignCorners(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapesForSizesCalcMode),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerSETest_NPU3720::getTestCaseName);

const std::vector<std::vector<float>> seInterpolateScalesElf = {{2.0f, 2.0f}};

auto seInterpolateParamsNearestElf = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesElf));
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_Nearest, InterpolateLayerSETest_NPU3720_ELF,
                         ::testing::Combine(seInterpolateParamsNearestElf(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerSETest_NPU3720_ELF::getTestCaseName);

auto seInterpolateParamsLinearElf = []() {
    return ::testing::Combine(::testing::Values(linearModes[1]), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesElf));
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_Linear, InterpolateLayerSETest_NPU3720_ELF,
                         ::testing::Combine(seInterpolateParamsLinearElf(), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(seInterpolateInputShapes),
                                            ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerSETest_NPU3720_ELF::getTestCaseName);

//
// Interpolate linear mode
//

auto interpolateCasesWithoutNearestLinearModeLargerNHWC = [](auto scales) {
    return ::testing::Combine(::testing::Values(linearModes[0]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(scales));
};
auto interpolateCasesWithoutNearestLinearModeLargerNCHW = [](auto scales) {
    return ::testing::Combine(::testing::Values(linearModes[0]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NCHWinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(linearInShapesLargeNCHW),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NCHWinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(linearInShapesLargeNCHW),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NHWCinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(linearInShapesLargeNHWC),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NHWCinput_NHWClayout_NCHWaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(linearInShapesLargeNHWC),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NCHWinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(linearInShapesLargeNCHW),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NCHWinput_NHWClayout_NCHWaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(linearInShapesLargeNCHW),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NHWCinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(linearInShapesLargeNHWC),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_linear_NHWCinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::ValuesIn(linearInShapesLargeNHWC),
                                            ::testing::ValuesIn(linearTargetShapesLarge),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

//
// Interpolate nearest align corner mode with tilling
//
const auto interpolateNearestAlignCorner = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(shapeCalculationMode), ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
        ::testing::ValuesIn(nearestModes), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateNearestTilingAlignCorner = ::testing::Combine(
        interpolateNearestAlignCorner, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 128, 170, 16})), ::testing::Values(std::vector<size_t>({256, 340})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_Nearest_Align_Corner, InterpolateLayerTest_NPU3720,
                         interpolateNearestTilingAlignCorner, InterpolateLayerTest_NPU3720::getTestCaseName);

// --------------------------------------------------
// ------ NPU3720 NoTiling Interpolate Testing ------
// --------------------------------------------------

const std::vector<std::vector<int64_t>> axesComplete = {{1, 2}, {2, 3}};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModeComplete = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const auto interpolateCasesNearestModeComplete = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(shapeCalculationMode), ::testing::ValuesIn(coordinateTransformModeComplete),
        ::testing::ValuesIn(nearestModes), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesComplete),
        ::testing::ValuesIn(defaultScales));
const auto interpolateParamsLinear = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(shapeCalculationMode), ::testing::ValuesIn(coordinateTransformModeComplete),
        ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesComplete),
        ::testing::ValuesIn(defaultScales));
const auto interpolateParamsLinearONNX = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(shapeCalculationMode), ::testing::ValuesIn(coordinateTransformModeComplete),
        ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes),
        ::testing::ValuesIn(defaultScales));
const auto interpolateParamsCubic = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(shapeCalculationMode), ::testing::ValuesIn(coordinateTransformModeComplete),
        ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesComplete),
        ::testing::ValuesIn(defaultScales));

const auto interpolateNearestNCHWUpscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 10, 30, 30})), ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNearestNHWCUpscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 10, 30, 30})), ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNearestNCHWDownscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 10, 40, 40})), ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNearestNHWCDownscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 10, 40, 40})), ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto interpolateLinearNCHWUpscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 10, 30, 30})), ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNHWCUpscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 10, 30, 30})), ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNCHWDownscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 10, 40, 40})), ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNHWCDownscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 10, 40, 40})), ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto interpolateLinearONNXNCHWUpscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 10, 30, 30})), ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearONNXNHWCUpscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 10, 30, 30})), ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearONNXNCHWDownscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 10, 40, 40})), ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearONNXNHWCDownscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 10, 40, 40})), ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

const auto interpolateCubicNCHWUpscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(std::vector<size_t>({1, 10, 30, 30})),
        ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateCubicNHWCUpscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(std::vector<size_t>({1, 10, 30, 30})),
        ::testing::Values(std::vector<size_t>({40, 40})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateCubicNCHWDownscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(std::vector<size_t>({1, 10, 40, 40})),
        ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateCubicNHWCDownscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(std::vector<size_t>({1, 10, 40, 40})),
        ::testing::Values(std::vector<size_t>({30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

// Mode NEAREST | Axes {1,2} & {2,3} | Coord Transform Mode: ALL | Nearest Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NCHW_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateNearestNCHWUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NHWC_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateNearestNHWCUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NCHW_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateNearestNCHWDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NHWC_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateNearestNHWCDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);

// Mode LINEAR | Axes {1,2} & {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NCHW_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NHWC_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NCHW_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NHWC_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);

// Mode LINEAR_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NCHW_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearONNXNCHWUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NHWC_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearONNXNHWCUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NCHW_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearONNXNCHWDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NHWC_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateLinearONNXNHWCDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);

// Mode CUBIC | Axes {1,2} & {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NCHW_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateCubicNCHWUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NHWC_Upscale, InterpolateLayerTest_NPU3720,
                         interpolateCubicNHWCUpscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NCHW_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateCubicNCHWDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NHWC_Downscale, InterpolateLayerTest_NPU3720,
                         interpolateCubicNHWCDownscale, InterpolateLayerTest_NPU3720::getTestCaseName);

//
// Optimize bilinear Interpolate with HALF_PIXEL and PYTORCH_HALF_PIXEL modes through the conversion to DW convolution
// and DMA
//

const std::vector<std::vector<size_t>> bilinearInterpolateToDwConvInputShapes = {
        {1, 40, 40, 40},
};

const std::vector<std::vector<size_t>> bilinearInterpolateToDwConvTargetShapes = {
        {80, 80},
        {120, 120},
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode>
        coordinateTransformModeHalfPixelandPytorchHalfPixel = {
                ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
};

auto bilinearInterpolateToDwConvParamsLinear = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModeHalfPixelandPytorchHalfPixel),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_bilinearInterpolateToDwConv, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(bilinearInterpolateToDwConvParamsLinear(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(bilinearInterpolateToDwConvInputShapes),
                                            ::testing::ValuesIn(bilinearInterpolateToDwConvTargetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);
//
// MapInterpolateOnDPU
//

const std::vector<std::vector<float>> mapBilinearInterpolateOnDPUScales = {{1.9444544315338135, 1.9444544315338135}};

const std::vector<std::vector<size_t>> mapBilinearInterpolateOnDPUInputShapes = {
        {1, 80, 72, 72},
};

const std::vector<std::vector<size_t>> mapBilinearInterpolateOnDPUTargetShapes = {
        {1, 80, 140, 140},
};

auto mapBilinearInterpolateOnDPUParamsLinear = []() {
    return ::testing::Combine(::testing::Values(linearModes[1]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModeComplete),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(allAxes), ::testing::ValuesIn(mapBilinearInterpolateOnDPUScales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_MapBilinearInterpolateOnDPU, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(mapBilinearInterpolateOnDPUParamsLinear(),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::ValuesIn(mapBilinearInterpolateOnDPUInputShapes),
                                            ::testing::ValuesIn(mapBilinearInterpolateOnDPUTargetShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

// --------------------------------------------------
// ------ NPU3720 Tiling Interpolate Testing ------
// --------------------------------------------------

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> interpolateAxes12ModeComplete = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
};

const auto interpolateParamsAxes12 = ::testing::Combine(
        ::testing::ValuesIn(interpolateAxes12ModeComplete), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(defaultScales));
const auto interpolateParamsAxes23 = ::testing::Combine(
        ::testing::ValuesIn(linearModes), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));

// UpScale| Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
const auto interpolateNCHWUpscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 4, 100, 180})), ::testing::Values(std::vector<size_t>({440, 550})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNCHWUpscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 1, 460, 620})), ::testing::Values(std::vector<size_t>({800, 1000})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNHWCUpscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 4, 99, 181})), ::testing::Values(std::vector<size_t>({440, 550})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNHWCUpscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 2, 190, 580})), ::testing::Values(std::vector<size_t>({500, 750})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

// UpScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
const auto interpolateLinearNCHWUpscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 3, 127, 540})), ::testing::Values(std::vector<size_t>({5, 317})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNCHWUpscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 3, 160, 520})), ::testing::Values(std::vector<size_t>({5, 300})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNHWCUpscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 2, 131, 630})), ::testing::Values(std::vector<size_t>({4, 317})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNHWCUpscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 2, 230, 400})), ::testing::Values(std::vector<size_t>({4, 500})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

// DownScale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and
// NHWC
const auto interpolateNCHWDownscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 4, 336, 640})), ::testing::Values(std::vector<size_t>({144, 256})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNCHWDownscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 1, 900, 700})), ::testing::Values(std::vector<size_t>({760, 520})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNHWCDownscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 4, 359, 639})), ::testing::Values(std::vector<size_t>({144, 256})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateNHWCDownscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 2, 600, 700})), ::testing::Values(std::vector<size_t>({230, 560})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

// DownScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
const auto interpolateLinearNCHWDownscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 5, 359, 640})), ::testing::Values(std::vector<size_t>({3, 143})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNCHWDownscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NCHW), ::testing::Values(InferenceEngine::Layout::NCHW),
        ::testing::Values(std::vector<size_t>({1, 5, 250, 620})), ::testing::Values(std::vector<size_t>({3, 160})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNHWCDownscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 4, 359, 630})), ::testing::Values(std::vector<size_t>({2, 143})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));
const auto interpolateLinearNHWCDownscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::NHWC), ::testing::Values(InferenceEngine::Layout::NHWC),
        ::testing::Values(std::vector<size_t>({1, 4, 600, 400})), ::testing::Values(std::vector<size_t>({2, 230})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()), ::testing::Values(additional_config));

// UpScale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNCHWUpscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
// Tracking number [E#88737]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Interpolate_Tiling_NCHW_Upscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNCHWUpscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNHWCUpscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNHWCUpscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// UpScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWUpscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWUpscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCUpscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCUpscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// DownScale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and
// NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNCHWDownscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNCHWDownscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNHWCDownscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNHWCDownscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// DownScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWDownscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWDownscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCDownscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCDownscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

}  // namespace
