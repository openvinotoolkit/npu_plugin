//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/normalize_l2.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class NormalizeL2LayerTestCommon :
        public NormalizeL2LayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class NormalizeL2LayerTest_NPU3700 : public NormalizeL2LayerTestCommon {};
class NormalizeL2LayerTest_NPU3720 : public NormalizeL2LayerTestCommon {};

TEST_P(NormalizeL2LayerTest_NPU3700, HW) {
    threshold = 0.04;
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(NormalizeL2LayerTest_NPU3720, HW) {
    threshold = 0.04;
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<ngraph::op::EpsMode> epsMode = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX,
};

/* ============= NPU 3700 ============= */
const std::vector<std::vector<int64_t>> axesNPU3700 = {
        {1},
};

// Values from real neural networks
// Contains eps > threshold. Incorrect kernel work, because the "max" mode isn't supported. [Track number: E#21695]
const std::vector<float> epsNPU3700 = {1.000000013351432e-10, 9.999999960041972e-13};

// [Track number: S#52943 E#85137]
std::vector<std::vector<size_t>> shapesNPU3700 = {{1, 128}, {1, 512}, {1, 8, 24, 64}, {1, 512, 64, 64}};

/* ============= NPU 3720 ============= */

const std::vector<float> eps = {
        9.9999999392252903e-09,
        1.000000013351432e-10,
        9.999999960041972e-13,
        9.9999998245167004e-14,
};

const std::vector<std::vector<int64_t>> axes2D = {{1}};
const std::vector<std::vector<int64_t>> axes3D = {{1}, {1, 2}, {0, 1, 2}};
const std::vector<std::vector<int64_t>> axes4D = {{1}, {1, 2}, {0, 1, 2}, {0, 1, 2, 3}};
const std::vector<std::vector<int64_t>> axesMini4D = {{1}, {1, 2}};
const std::vector<std::vector<int64_t>> axesTiling4D = {{1}, {2}, {3}, {1, 2}};

std::vector<std::vector<size_t>> shapes2D = {
        {1, 128},
        {1, 256},
        {1, 512},
};

std::vector<std::vector<size_t>> shapes3D = {{1, 5, 3}, {1, 20, 200}};

std::vector<std::vector<size_t>> shapes4D = {
        {1, 8, 24, 64},   {1, 1024, 1, 1},  {1, 128, 50, 85}, {1, 512, 64, 64},
        {1, 512, 40, 40}, {1, 512, 20, 20}, {1, 512, 38, 38}, {1, 128, 25, 43},
};

std::vector<std::vector<size_t>> shapesTiling4D = {{1, 512, 36, 36}, {1, 512, 37, 37}, {1, 512, 44, 43}};

const auto params =
        testing::Combine(testing::ValuesIn(axesNPU3700), testing::ValuesIn(epsNPU3700), testing::ValuesIn(epsMode),
                         testing::ValuesIn(shapesNPU3700), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params2D = testing::Combine(testing::ValuesIn(axes2D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(shapes2D), testing::ValuesIn(netPrecisions),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params3D = testing::Combine(testing::ValuesIn(axes3D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(shapes3D), testing::ValuesIn(netPrecisions),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params4D = testing::Combine(testing::ValuesIn(axes4D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(shapes4D), testing::ValuesIn(netPrecisions),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsMini4D =
        testing::Combine(testing::ValuesIn(axesMini4D), testing::Values(eps[0]), testing::Values(epsMode[0]),
                         testing::ValuesIn(shapes4D), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

/* ============= NPU 3700 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2, NormalizeL2LayerTest_NPU3700, params,
                         NormalizeL2LayerTest_NPU3700::getTestCaseName);

/* ============= NPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_2D, NormalizeL2LayerTest_NPU3720, params2D,
                         NormalizeL2LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_3D, NormalizeL2LayerTest_NPU3720, params3D,
                         NormalizeL2LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_4D, NormalizeL2LayerTest_NPU3720, params4D,
                         NormalizeL2LayerTest_NPU3720::getTestCaseName);

/* ============= Tiling ============= */

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_1, NormalizeL2LayerTest_NPU3720,
                         testing::Combine(testing::ValuesIn(axesTiling4D),
                                          testing::ValuesIn(std::vector<float>{3.0815954528967052E-41}),
                                          testing::Values(epsMode[0]), testing::ValuesIn(shapesTiling4D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         NormalizeL2LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_2, NormalizeL2LayerTest_NPU3720,
                         testing::Combine(testing::Values(std::vector<int64_t>({3})),
                                          testing::ValuesIn(std::vector<float>{9.9999997473787516e-05}),
                                          testing::Values(epsMode[1]),
                                          testing::Values(std::vector<size_t>({3, 3, 64, 2304})),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         NormalizeL2LayerTest_NPU3720::getTestCaseName);

}  // namespace
