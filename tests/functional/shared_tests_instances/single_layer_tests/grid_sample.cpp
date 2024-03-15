//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/grid_sample.hpp"
#include <common/functions.h>
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class GridSampleLayerTestCommon : public GridSampleLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        GridSampleLayerTest::SetUp();
    }

    void GenerateInputs() override {
        inputs.clear();
        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (size_t i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
            InferenceEngine::InputInfo::CPtr info = infoIt->second;
            auto blob = GenerateInput(*info);
            if (i > 0) {
                blob = FuncTestUtils::createAndFillBlobFloatNormalDistribution(info->getTensorDesc(), 0, 0.5);
            } else {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), 10, 0);
            }
            inputs.push_back(blob);
        }
    }
};

class GridSampleLayerTest_NPU3720 : public GridSampleLayerTestCommon {};

TEST_P(GridSampleLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using GridSampleOp = ov::op::v9::GridSample;

namespace {

const std::vector<std::vector<size_t>> dataShapes = {{2, 2, 3, 4}};

const std::vector<std::vector<size_t>> gridShapes = {{2, 2, 3, 2}};

const std::vector<std::vector<size_t>> dataShapesTiling = {{1, 2, 800, 800}};

const std::vector<std::vector<size_t>> gridShapesTiling = {{1, 2, 2, 2}};

const std::vector<bool> alignCorners = {true, false};

const std::vector<GridSampleOp::InterpolationMode> modes = {
        GridSampleOp::InterpolationMode::BILINEAR,
        GridSampleOp::InterpolationMode::NEAREST,
        GridSampleOp::InterpolationMode::BICUBIC,
};

const std::vector<GridSampleOp::PaddingMode> paddingModes = {
        GridSampleOp::PaddingMode::ZEROS, GridSampleOp::PaddingMode::BORDER, GridSampleOp::PaddingMode::REFLECTION};

const std::vector<InferenceEngine::Precision> dataPrecisions = {
        InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> gridPrecisions = {
        InferenceEngine::Precision::FP16,
};

const auto params = testing::Combine(
        ::testing::ValuesIn(dataShapes), ::testing::ValuesIn(gridShapes), ::testing::ValuesIn(alignCorners),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataPrecisions),
        ::testing::ValuesIn(gridPrecisions), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsTiling = testing::Combine(
        ::testing::ValuesIn(dataShapesTiling), ::testing::ValuesIn(gridShapesTiling), ::testing::ValuesIn(alignCorners),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataPrecisions),
        ::testing::ValuesIn(gridPrecisions), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// NPU3720
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample, GridSampleLayerTest_NPU3720, params,
                         GridSampleLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample_Tiling, GridSampleLayerTest_NPU3720, paramsTiling,
                         GridSampleLayerTest::getTestCaseName);

}  // namespace
