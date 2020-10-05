// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbInterpolateLayerTest: public InterpolateLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbInterpolateLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
    {1, 1, 30, 30},
};

const std::vector<std::vector<size_t>> targetShapes = {
    {1, 1, 40, 40},
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
    ngraph::op::v4::Interpolate::InterpolateMode::linear,
    ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
    ngraph::op::v4::Interpolate::InterpolateMode::cubic,
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
    ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
    ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
    ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
    ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
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

const std::vector<std::vector<size_t>> pads = {
    {0, 0, 1, 1},
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

const auto interpolateCasesWithoutNearest = ::testing::Combine(
    ::testing::ValuesIn(modesWithoutNearest),
    ::testing::ValuesIn(shapeCalculationMode),
    ::testing::ValuesIn(coordinateTransformModes),
    ::testing::ValuesIn(defaultNearestMode),
    ::testing::ValuesIn(antialias),
    ::testing::ValuesIn(pads),
    ::testing::ValuesIn(pads),
    ::testing::ValuesIn(cubeCoefs),
    ::testing::ValuesIn(defaultAxes),
    ::testing::ValuesIn(defaultScales));

const auto interpolateCases = ::testing::Combine(
    ::testing::ValuesIn(nearestMode),
    ::testing::ValuesIn(shapeCalculationMode),
    ::testing::ValuesIn(coordinateTransformModes),
    ::testing::ValuesIn(nearestModes),
    ::testing::ValuesIn(antialias),
    ::testing::ValuesIn(pads),
    ::testing::ValuesIn(pads),
    ::testing::ValuesIn(cubeCoefs),
    ::testing::ValuesIn(defaultAxes),
    ::testing::ValuesIn(defaultScales));

// Tests are disabled due to error:
// C++ exception with description "Cannot convert layer "Interpolate_xxxxx" due to unsupported layer type "Interpolate"
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:138
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#39978]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Interpolate_Basic, KmbInterpolateLayerTest, ::testing::Combine(
    interpolateCasesWithoutNearest,
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(targetShapes),
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        KmbInterpolateLayerTest::getTestCaseName);

// Tests are disabled due to error:
// C++ exception with description "Cannot convert layer "Interpolate_xxxxx" due to unsupported layer type "Interpolate"
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:138
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#39978]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Interpolate_Nearest, KmbInterpolateLayerTest, ::testing::Combine(
    interpolateCases,
    ::testing::ValuesIn(netPrecisions),
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(targetShapes),
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        KmbInterpolateLayerTest::getTestCaseName);

} // namespace
