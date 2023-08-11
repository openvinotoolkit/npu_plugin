//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXLrnLayerTest : public LrnLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXLrnLayerTest_VPU3700 : public VPUXLrnLayerTest {};
class VPUXLrnLayerTest_VPU3720 : public VPUXLrnLayerTest {};

TEST_P(VPUXLrnLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXLrnLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
// Common params

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3}};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck, VPUXLrnLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                                            ::testing::Values(size), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(std::vector<size_t>({1, 10, 3, 2})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXLrnLayerTest_VPU3700::getTestCaseName);

const std::vector<std::vector<int64_t>> axes_vpu3720 = {{1}, {2}, {1, 2}, {2, 3}, {1, 2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnCheck, VPUXLrnLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                                            ::testing::Values(size), ::testing::ValuesIn(axes_vpu3720),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(std::vector<size_t>({1, 10, 3, 2})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXLrnLayerTest_VPU3720::getTestCaseName);

}  // namespace
