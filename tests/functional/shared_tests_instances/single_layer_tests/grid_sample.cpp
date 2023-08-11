//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/grid_sample.hpp"
#include <common/functions.h>
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXGridSampleLayerTest : public GridSampleLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
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

class VPUXGridSampleLayerTest_VPU3720 : public VPUXGridSampleLayerTest {};

TEST_P(VPUXGridSampleLayerTest_VPU3720, HW) {
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

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample_VPU3720, VPUXGridSampleLayerTest_VPU3720,
                         testing::Combine(::testing::ValuesIn(dataShapes), ::testing::ValuesIn(gridShapes),
                                          ::testing::ValuesIn(alignCorners), ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataPrecisions),
                                          ::testing::ValuesIn(gridPrecisions),
                                          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         GridSampleLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample_VPU3720_Tiling, VPUXGridSampleLayerTest_VPU3720,
                         testing::Combine(::testing::ValuesIn(dataShapesTiling), ::testing::ValuesIn(gridShapesTiling),
                                          ::testing::ValuesIn(alignCorners), ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataPrecisions),
                                          ::testing::ValuesIn(gridPrecisions),
                                          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         GridSampleLayerTest::getTestCaseName);

}  // namespace
