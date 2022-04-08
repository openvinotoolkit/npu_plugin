//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/grid_sample.hpp"
#include <common/functions.h>
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class KmbGridSampleLayerTest_VPU3720 :
        public GridSampleLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};
TEST_P(KmbGridSampleLayerTest_VPU3720, MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using GridSampleOp = ov::op::v9::GridSample;

namespace {

const std::vector<std::vector<size_t>> dataShapes = {
        {1, 2, 3, 4},
        {1, 2, 3, 3},
};

const std::vector<std::vector<size_t>> gridShapes = {
        {1, 1, 3, 2},
        {1, 2, 3, 2},
};

const std::vector<std::vector<size_t>> dataShapesTiling = {{1, 2, 800, 800}};

const std::vector<std::vector<size_t>> gridShapesTiling = {{1, 2, 2, 2}};

const std::vector<bool> alignCorners = {true};

const std::vector<GridSampleOp::InterpolationMode> modes = {
        GridSampleOp::InterpolationMode::BILINEAR,
};

const std::vector<GridSampleOp::PaddingMode> paddingModes = {
        GridSampleOp::PaddingMode::BORDER,
};

const std::vector<InferenceEngine::Precision> dataPrecisions = {
        InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> gridPrecisions = {
        InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample_VPU3720, KmbGridSampleLayerTest_VPU3720,
                         testing::Combine(::testing::ValuesIn(dataShapes), ::testing::ValuesIn(gridShapes),
                                          ::testing::ValuesIn(alignCorners), ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataPrecisions),
                                          ::testing::ValuesIn(gridPrecisions),
                                          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         GridSampleLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GridSample_VPU3720_Tiling, KmbGridSampleLayerTest_VPU3720,
                         testing::Combine(::testing::ValuesIn(dataShapesTiling), ::testing::ValuesIn(gridShapesTiling),
                                          ::testing::ValuesIn(alignCorners), ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataPrecisions),
                                          ::testing::ValuesIn(gridPrecisions),
                                          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         GridSampleLayerTest::getTestCaseName);

}  // namespace
