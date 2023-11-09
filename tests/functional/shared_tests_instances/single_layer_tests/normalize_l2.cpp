//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/normalize_l2.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXNormalizeL2LayerTest : public NormalizeL2LayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class VPUXNormalizeL2LayerTest_VPU3700 : public VPUXNormalizeL2LayerTest {};
class VPUXNormalizeL2LayerTest_VPU3720 : public VPUXNormalizeL2LayerTest {};

TEST_P(VPUXNormalizeL2LayerTest_VPU3700, HW) {
    threshold = 0.04;
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXNormalizeL2LayerTest_VPU3720, HW) {
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

/* ============= VPU 3700 ============= */
const std::vector<std::vector<int64_t>> axesVPU3700 = {
        {1},
};

// Values from real neural networks
// Contains eps > threshold. Incorrect kernel work, because the "max" mode isn't supported. [Track number: E#21695]
const std::vector<float> epsVPU3700 = {1.000000013351432e-10, 9.999999960041972e-13};

// [Track number: S#52943 E#85137]
std::vector<std::vector<size_t>> shapesVPU3700 = {{1, 128}, {1, 512}, {1, 8, 24, 64}, {1, 512, 64, 64}};

/* ============= VPU 3720 ============= */

const std::vector<float> eps = {9.9999999392252903e-09,
                                1.000000013351432e-10,
                                9.999999960041972e-13,
                                9.9999998245167004e-14,
                                0.0001,
                                0.001,
                                0.5};

const std::vector<float> eps_precommit = {
        9.9999999392252903e-09,
        0.0001,
};

const std::vector<std::vector<int64_t>> axes2D = {{1}};
const std::vector<std::vector<int64_t>> axes3D = {{1}, {1, 2}, {0, 1, 2}};
const std::vector<std::vector<int64_t>> axes4D = {{1}, {1, 2}, {0, 1, 2}, {0, 1, 2, 3}};

std::vector<std::vector<size_t>> shapes2D = {
        {1, 128},
        {1, 256},
        {1, 512},
};

std::vector<std::vector<size_t>> shapes3D = {{1, 5, 3}, {1, 20, 200}};

std::vector<std::vector<size_t>> shapes4D = {
        {1, 8, 24, 64}, {1, 128, 25, 43}, {1, 3, 10, 5}, {1, 1, 1, 10},

        // TODO: Kindly test this once tiling is enabled
        // {1, 128, 50, 85},
        // {1, 512, 64, 64}
        // {1, 512, 40, 40},
        // {1, 512, 20, 20},
        // {1, 512, 38, 38},
        // {1, 128, 25, 43},
};

const auto params =
        testing::Combine(testing::ValuesIn(axesVPU3700), testing::ValuesIn(epsVPU3700), testing::ValuesIn(epsMode),
                         testing::ValuesIn(shapesVPU3700), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params2D = testing::Combine(testing::ValuesIn(axes2D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(shapes2D), testing::ValuesIn(netPrecisions),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params2D_precommit =
        testing::Combine(testing::ValuesIn(axes2D), testing::ValuesIn(eps_precommit), testing::ValuesIn(epsMode),
                         testing::ValuesIn(shapes2D), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params3D = testing::Combine(testing::ValuesIn(axes3D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(shapes3D), testing::ValuesIn(netPrecisions),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params4D = testing::Combine(testing::ValuesIn(axes4D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(shapes4D), testing::ValuesIn(netPrecisions),
                                       testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

/* ============= VPU 3700 ============= */

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_NormalizeL2, VPUXNormalizeL2LayerTest_VPU3700, params,
                         VPUXNormalizeL2LayerTest_VPU3700::getTestCaseName);

/* ============= VPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_2D, VPUXNormalizeL2LayerTest_VPU3720, params2D,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_3D, VPUXNormalizeL2LayerTest_VPU3720, params3D,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_4D, VPUXNormalizeL2LayerTest_VPU3720, params4D,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

/* ============= Tiling ============= */

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_1, VPUXNormalizeL2LayerTest_VPU3720,
                         testing::Combine(testing::Values(std::vector<int64_t>({1})),
                                          testing::ValuesIn(std::vector<float>{3.0815954528967052E-41}),
                                          testing::Values(epsMode[0]),
                                          testing::Values(std::vector<size_t>({1, 512, 37, 37})),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_2, VPUXNormalizeL2LayerTest_VPU3720,
                         testing::Combine(testing::Values(std::vector<int64_t>({3})),
                                          testing::ValuesIn(std::vector<float>{9.9999997473787516e-05}),
                                          testing::Values(epsMode[1]),
                                          testing::Values(std::vector<size_t>({3, 3, 64, 2304})),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

}  // namespace
