// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/log_softmax.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXLogSoftmaxLayerTest : public LogSoftmaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXLogSoftmaxLayerTest_VPU3700 : public VPUXLogSoftmaxLayerTest {};
class VPUXLogSoftmaxLayerTest_VPU3720 : public VPUXLogSoftmaxLayerTest {};

// Disabled as 'convert-subtract-to-negative-add' pass is not ready for one/more platforms in `ReferenceSW` mode
// These tests shall be re-enabled and revalidate once such pass is added to 'ReferenceSW' pipeline
TEST_P(VPUXLogSoftmaxLayerTest_VPU3700, DISABLED_SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}
TEST_P(VPUXLogSoftmaxLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXLogSoftmaxLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecision = {InferenceEngine::Precision::FP16};

const std::vector<InferenceEngine::Precision> dataPrecision = {InferenceEngine::Precision::UNSPECIFIED};

/* ============= 2D LogSoftmax ============= */

const std::vector<InferenceEngine::Layout> layouts2D = {InferenceEngine::Layout::NC};

std::vector<std::vector<size_t>> inShapes2D = {
        {12, 5}, {1200, 5}  // real case
};

std::vector<int64_t> axis2D = {0, 1};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts2D), testing::ValuesIn(layouts2D), testing::ValuesIn(inShapes2D),
        testing::ValuesIn(axis2D), testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, VPUXLogSoftmaxLayerTest_VPU3700, params2D,
                         VPUXLogSoftmaxLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_LogSoftmax_2D, VPUXLogSoftmaxLayerTest_VPU3720, params2D,
                         VPUXLogSoftmaxLayerTest_VPU3720::getTestCaseName);

/* ============= 3D/4D LogSoftmax ============= */

const std::vector<InferenceEngine::Layout> layouts = {InferenceEngine::Layout::ANY};
//
// [Track number: E#37278]
//
std::vector<std::vector<size_t>> inShapes = {
        {1, 20, 256, 512}, {1, 10, 256, 512},
        // {5, 30, 1}
};

std::vector<int64_t> axis = {2, 3};
//
// [Track number: E#37278]
//
//     std::vector<int64_t> axis = {0, 1, 2, 3};
//     std::vector<int64_t> axis = {0, 1, 2};

const auto params = testing::Combine(testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision),
                                     testing::ValuesIn(dataPrecision), testing::ValuesIn(layouts),
                                     testing::ValuesIn(layouts), testing::ValuesIn(inShapes), testing::ValuesIn(axis),
                                     testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                     ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_3D_4D, VPUXLogSoftmaxLayerTest_VPU3700, params,
                         VPUXLogSoftmaxLayerTest_VPU3700::getTestCaseName);

std::vector<std::vector<size_t>> inShapesVPU3720 = {
        {1, 10, 7, 4},
        {1, 2, 204, 62},
        {3, 20, 1, 15},
};

std::vector<int64_t> axisVPU3720 = {0, 2};

const auto paramsVPU3720 = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts), testing::ValuesIn(layouts), testing::ValuesIn(inShapesVPU3720),
        testing::ValuesIn(axisVPU3720), testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(precommit_LogSoftmax_3D_4D, VPUXLogSoftmaxLayerTest_VPU3720, paramsVPU3720,
                         VPUXLogSoftmaxLayerTest_VPU3720::getTestCaseName);

const auto paramsTiling = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts), testing::ValuesIn(layouts), testing::Values(std::vector<size_t>{1, 48, 160, 80}),
        testing::Values(1), testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(precommit_LogSoftmax_3D_4D_tiling_VPU3720, VPUXLogSoftmaxLayerTest_VPU3720, paramsTiling,
                         VPUXLogSoftmaxLayerTest_VPU3720::getTestCaseName);

}  // namespace
