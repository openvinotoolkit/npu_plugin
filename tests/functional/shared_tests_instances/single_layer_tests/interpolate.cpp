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

class KmbInterpolate1Test: public Interpolate1LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

public:
// Reference version for Interpolate-1 layer doesn't implemented in OpenVino.
// It is enough to check the conversion of Interpolate-1 layer to Interpolate-4 layer,
// as test for Interpolate-4 has already exist. So skip Validate step.
void Run() override {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::cout << "KmbLayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
    LayerTestsUtils::KmbTestReport& report = LayerTestsUtils::KmbTestReport::getInstance();
    const auto& testInfo = testing::UnitTest::GetInstance()->current_test_info();
    report.run(testInfo);

    try {
        if (envConfig.IE_KMB_TESTS_RUN_COMPILER) {
            std::cout << "KmbLayerTestsCommon::Compile" << std::endl;
            std::ostringstream ostr;
            ostr << "LoadNetwork Config: ";
            for (const auto& item : configuration) {
                ostr << item.first << "=" << item.second << "; ";
            }
            std::cout << ostr.str() << std::endl;
            SkipBeforeLoad();

            ASSERT_NO_THROW(executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration));
            report.compiled(testInfo);

            if (envConfig.IE_KMB_TESTS_RUN_EXPORT) {
                std::cout << "KmbLayerTestsCommon::ExportNetwork()" << std::endl;
                ASSERT_NO_THROW(ExportNetwork());
            }
        } else {
            std::cout << "KmbLayerTestsCommon::ImportNetwork()" << std::endl;
            SkipBeforeLoad();
            SkipBeforeImport();
            ImportNetwork();
            report.imported(testInfo);
        }
        GenerateInputs();
        if (envConfig.IE_KMB_TESTS_EXPORT_INPUT) {
            std::cout << "KmbLayerTestsCommon::ExportInput()" << std::endl;
            ExportInput();
        }
        if (envConfig.IE_KMB_TESTS_IMPORT_INPUT) {
            std::cout << "KmbLayerTestsCommon::ImportInput()" << std::endl;
            ImportInput();
        }
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            std::cout << "KmbLayerTestsCommon::Infer()" << std::endl;
            SkipBeforeInfer();
            Infer();
            report.inferred(testInfo);
        }
        if (envConfig.IE_KMB_TESTS_EXPORT_OUTPUT) {
            std::cout << "KmbLayerTestsCommon::ExportOutput()" << std::endl;
            ExportOutput();
        }
        //skip Validate step
        std::cout << "Skip KmbLayerTestsCommon::Validate()" << std::endl;
    } catch (const LayerTestsUtils::KmbSkipTestException& e) {
        std::cout << "Skipping the test due to: " << e.what() << std::endl;
        report.skipped(testInfo);
        GTEST_SKIP() << "Skipping the test due to: " << e.what();
    }
}
};

TEST_P(KmbInterpolate1Test, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
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

const auto interpolateCasesNearestMode = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesNearest),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesWithoutNearestMode = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_mode, KmbInterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearestMode,
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
        interpolateCasesWithoutNearestMode,
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

} // namespace
