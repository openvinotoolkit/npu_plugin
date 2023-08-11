//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <base/behavior_test_utils.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "vpux_private_config.hpp"

using CompileForDifferentPlatformsTests = BehaviorTestsUtils::BehaviorTestsBasic;
namespace {

// [Track number: E#15711]
// [Track number: E#15635]
TEST_P(CompileForDifferentPlatformsTests, CompilationForSpecificPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
        auto cfg = configuration;
        cfg["VPUX_CREATE_EXECUTOR"] = "0";
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, target_device, cfg));
    }
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

// TODO Remove deprecated platform names with VPU prefix in future releases
const std::vector<std::map<std::string, std::string>> configs = {
        {},
        {{VPUX_CONFIG_KEY(PLATFORM), "3400"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3700"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3800"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3900"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3720"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3400"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3700"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3800"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3900"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3720"}},
        {{CONFIG_KEY(DEVICE_ID), "3400"}},
        {{CONFIG_KEY(DEVICE_ID), "3700"}},
        {{CONFIG_KEY(DEVICE_ID), "3800"}},
        {{CONFIG_KEY(DEVICE_ID), "3900"}},
        {{CONFIG_KEY(DEVICE_ID), "3720"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, CompileForDifferentPlatformsTests,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         CompileForDifferentPlatformsTests::getTestCaseName);
}  // namespace
