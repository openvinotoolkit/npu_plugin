//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/normalize_l2.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXNormalizeL2LayerTest : public NormalizeL2LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
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

//
// If shape N-dimensional and axes contains N-1 dims - tests failed. [Track number: E#21695]
//
const std::vector<std::vector<int64_t>> axes = {
        // {}, 16900: NormalizeL2 output mismatch for empty axes case
        {1},
        //{1,2}
        // Only for shapes = { ., ., .}
        //{0, 1, 2},
        //{0, 1, 2, 3}
};

// Values from real neural networks
const std::vector<float> eps = {1.000000013351432e-10, 9.999999960041972e-13};

//
// Contains eps > threshold. Incorrect kernel work, because the "max" mode isn't supported. [Track number: E#21695]
//
// const std::vector<float> eps = {0.0001, 0.001, 0.5};

const std::vector<ngraph::op::EpsMode> epsMode = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX,
};

// There is a error at validation step for shapes which size are not equal to 4.
// Possibly it is error in run-time due to only 4D shapes are allowed.
// Example of output on board:
// ...
// TestReportProgress: KmbNormalizeL2LayerTest inferred
// KmbLayerTestsCommon::Validate()
// LayerTestsCommon::Validate()
// openvino/inference-engine/tests/functional/shared_test_classes/include/shared_test_classes/
// base/layer_test_utils.hpp:173: Failure
// Value of: max != 0 && (diff <= static_cast<float>(threshold))
// Actual: false
// Expected: true
// Relative comparison of values expected: 0 and actual: nan at index 0 with threshold 0.0099999997764825821 failed
// [Track number: S#52943]

//
//[Track number: E#21695]
//
std::vector<std::vector<size_t>> shapes = {
        {1, 128},
        {1, 512},
        {1, 8, 24, 64},
        //{1, 3, 10, 5},

        // Turn off the axes = {0, 1, 2, 3}
        // Incorrect kernel work in case axes = {1}
        //
        //{1, 5, 3}

        // Values from real neural networks
        //{1, 512, 40, 40},
        //{1, 512, 20, 20},
        {1, 512, 64, 64}
        //{1, 512, 38, 38},
        //{1, 128, 25, 43},
        //{1, 128, 50, 85}

        // Incorrect kernel work
        //{1, 1, 1, 10}
};

const std::vector<std::vector<int64_t>> axesVPUX = {{1},

                                                    // Only valid for shapes with 3D and above
                                                    {1, 2},
                                                    {0, 1, 2},

                                                    // Only valid for shapes with 4D and above
                                                    {0, 1, 2, 3}};

const std::vector<float> epsVPUX = {9.9999999392252903e-09, 1.000000013351432e-10, 9.999999960041972e-13,
                                    9.9999998245167004e-14};

std::vector<std::vector<size_t>> shapesVPUX = {
        {1, 128},
        {1, 8, 24, 64},

        // Values from real neural networks
        {1, 256},
        {1, 512},
        {1, 128, 25, 43},
        // TODO: Kindly test this once tiling is enabled
        // {1, 128, 50, 85},
        // {1, 512, 64, 64}
};

const auto params = testing::Combine(testing::ValuesIn(axes), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                     testing::ValuesIn(shapes), testing::ValuesIn(netPrecisions),
                                     testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto paramsVPUX =
        testing::Combine(testing::ValuesIn(axesVPUX), testing::Values(epsVPUX[0]), testing::Values(epsMode[0]),
                         testing::Values(shapesVPUX[1]), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto paramsVPUXRealNet_Set1 =
        testing::Combine(testing::Values(axesVPUX[0]), testing::Values(epsVPUX[3]), testing::Values(epsMode[1]),
                         testing::Values(shapesVPUX[2]), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto paramsVPUXRealNet_Set2 =
        testing::Combine(testing::Values(axesVPUX[0]), testing::Values(epsVPUX[1]), testing::Values(epsMode[1]),
                         testing::Values(shapesVPUX[3]), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto paramsVPUXRealNet_Set3 =
        testing::Combine(testing::Values(axesVPUX[0]), testing::Values(epsVPUX[0]), testing::Values(epsMode[0]),
                         testing::Values(shapesVPUX[4]), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto paramsVPUXPrecommit =
        testing::Combine(testing::Values(axesVPUX[0]), testing::Values(epsVPUX[2]), testing::ValuesIn(epsMode),
                         testing::Values(shapesVPUX[0]), testing::ValuesIn(netPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_NormalizeL2, VPUXNormalizeL2LayerTest_VPU3700, params,
                         VPUXNormalizeL2LayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2, VPUXNormalizeL2LayerTest_VPU3720, paramsVPUX,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_Set1, VPUXNormalizeL2LayerTest_VPU3720, paramsVPUXRealNet_Set1,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_Set2, VPUXNormalizeL2LayerTest_VPU3720, paramsVPUXRealNet_Set2,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_Set3, VPUXNormalizeL2LayerTest_VPU3720, paramsVPUXRealNet_Set3,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_NormalizeL2, VPUXNormalizeL2LayerTest_VPU3720, paramsVPUXPrecommit,
                         VPUXNormalizeL2LayerTest_VPU3720::getTestCaseName);

}  // namespace
