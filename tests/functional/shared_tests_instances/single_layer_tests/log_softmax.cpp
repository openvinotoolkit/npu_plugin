// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/log_softmax.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class LogSoftmaxLayerTestCommon : public LogSoftmaxLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class LogSoftmaxLayerTest_NPU3700 : public LogSoftmaxLayerTestCommon {};
class LogSoftmaxLayerTest_NPU3720 : public LogSoftmaxLayerTestCommon {};

// Disabled as 'convert-subtract-to-negative-add' pass is not ready for one/more platforms in `ReferenceSW` mode
// These tests shall be re-enabled and revalidate once such pass is added to 'ReferenceSW' pipeline
TEST_P(LogSoftmaxLayerTest_NPU3700, DISABLED_SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}
TEST_P(LogSoftmaxLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(LogSoftmaxLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecision = {InferenceEngine::Precision::FP16};
const std::vector<InferenceEngine::Precision> dataPrecision = {InferenceEngine::Precision::UNSPECIFIED};

const std::vector<InferenceEngine::Layout> layouts = {InferenceEngine::Layout::ANY};
const std::vector<InferenceEngine::Layout> layouts2D = {InferenceEngine::Layout::NC};

/* ============= 2D/3D/4D LogSoftmax NPU3700 ============= */

std::vector<std::vector<size_t>> inShapes2D_NPU3700 = {
        {12, 5}, {1200, 5}  // real case
};

// Tracking number [E#85137]
std::vector<std::vector<size_t>> inShapes4D_NPU3700 = {
        {1, 20, 256, 512},
        {1, 10, 256, 512},
};

std::vector<int64_t> axis = {2, 3};
std::vector<int64_t> axis2D = {0, 1};

const auto params2D_NPU3700 = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts2D), testing::ValuesIn(layouts2D), testing::ValuesIn(inShapes2D_NPU3700),
        testing::ValuesIn(axis2D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(std::map<std::string, std::string>({})));

const auto params3D_4D_NPU3700 = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts), testing::ValuesIn(layouts), testing::ValuesIn(inShapes4D_NPU3700),
        testing::ValuesIn(axis), testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, LogSoftmaxLayerTest_NPU3700, params2D_NPU3700,
                         LogSoftmaxLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_3D_4D, LogSoftmaxLayerTest_NPU3700, params3D_4D_NPU3700,
                         LogSoftmaxLayerTest_NPU3700::getTestCaseName);

/* ============= LogSoftmax NPU3720 ============= */

std::vector<std::vector<size_t>> inShapes2D = {{12, 5},    {1, 40},   {1, 66},   {1, 72},   {5, 120},
                                               {5, 59},    {64, 29},  {1, 2312}, {1, 4192}, {1, 4335},
                                               {10, 6495}, {1200, 5}, {2708, 7}};

std::vector<std::vector<size_t>> inShapes3D = {{5, 30, 1}};

std::vector<std::vector<size_t>> inShapes4D = {
        {1, 10, 7, 4},
        {1, 2, 204, 62},
        {3, 20, 1, 15},
};

std::vector<int64_t> axis3D = {0, 1, 2};
std::vector<int64_t> axis4D = {0, 1, 2, 3};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts2D), testing::ValuesIn(layouts2D), testing::ValuesIn(inShapes2D),
        testing::ValuesIn(axis2D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(std::map<std::string, std::string>({})));

const auto params3D = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts), testing::ValuesIn(layouts), testing::ValuesIn(inShapes3D),
        testing::ValuesIn(axis3D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(std::map<std::string, std::string>({})));

const auto params4D = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts), testing::ValuesIn(layouts), testing::ValuesIn(inShapes4D),
        testing::ValuesIn(axis4D), testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(std::map<std::string, std::string>({})));

const auto paramsTiling = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts), testing::ValuesIn(layouts), testing::Values(std::vector<size_t>{1, 48, 160, 80}),
        testing::Values(1), testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(std::map<std::string, std::string>({})));

/* ============= LogSoftmax NPU3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, LogSoftmaxLayerTest_NPU3720, params2D,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_3D, LogSoftmaxLayerTest_NPU3720, params3D,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_4D, LogSoftmaxLayerTest_NPU3720, params4D,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_tiling, LogSoftmaxLayerTest_NPU3720, paramsTiling,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);

}  // namespace
