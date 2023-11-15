//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/power.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXPowerLayerTest_VPU3700 : public PowerLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SkipBeforeLoad() override {
        if (envConfig.IE_NPU_TESTS_RUN_INFER) {
            throw LayerTestsUtils::VpuSkipTestException("layer test networks hang the board");
        }
    }
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::VpuSkipTestException("comparison fails");
    }
};

TEST_P(VPUXPowerLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {{{1, 8}},   {{2, 16}},  {{3, 32}},  {{4, 64}},
                                                          {{5, 128}}, {{6, 256}}, {{7, 512}}, {{8, 1024}}};

std::vector<std::vector<float>> Power = {
        {0.0f}, {0.5f}, {1.0f}, {1.1f}, {1.5f}, {2.0f},
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_power, VPUXPowerLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::ValuesIn(Power)),
                         VPUXPowerLayerTest_VPU3700::getTestCaseName);

}  // namespace
