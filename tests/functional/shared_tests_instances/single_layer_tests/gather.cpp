// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include <vpux/vpux_plugin_config.hpp>

namespace LayerTestsDefinitions {

class KmbGatherLayerTest: public GatherLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbGatherLayerTest, CompareWithRefs) {
   // Enable NCHW layout
    core->SetConfig({{VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT), CONFIG_VALUE(YES)}},
                      LayerTestsUtils::testPlatformTargetDevice);
    Run();
}

TEST_P(KmbGatherLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
        //std::vector<size_t>{10, 20, 30, 40},
          std::vector<size_t>{ 5,  6,  7,  8},
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes = {
        std::vector<size_t>{4},
        // std::vector<size_t>{2, 2}  //  Only 1D shape for indices is supported
};

const std::vector<int> axes = {0, 1, 2, 3, /*-1*/};  // Only positive axis value is supported

const auto params = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
        smoke_Gather,
        KmbGatherLayerTest,
        params,
        KmbGatherLayerTest::getTestCaseName
);

}  // namespace
