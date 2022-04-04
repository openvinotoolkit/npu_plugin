// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "vpux_private_config.hpp"
#include "common/functions.h"

using CompileForDifferentPlatformsTests = BehaviorTestsUtils::BehaviorTestsBasic;
namespace {

// [Track number: E#15711]
// [Track number: E#15635]
TEST_P(CompileForDifferentPlatformsTests, CompilationForSpecificPlatform) {
    if (getBackendName(*ie) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
#if defined(__arm__) || defined(__aarch64__)
    GTEST_SKIP() << "Compilation only";
#endif
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
        configuration[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MCM);
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

// TODO Remove deprecated platform names with VPU prefix in future releases
const std::vector<std::map<std::string, std::string>> configs = {
        {{VPUX_CONFIG_KEY(PLATFORM), "3400_A0"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3400"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3700"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3800"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "3900"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "37XX"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3400_A0"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3400"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3700"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3800"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3900"}},
        {{VPUX_CONFIG_KEY(PLATFORM), "VPUX37XX"}},
        {{CONFIG_KEY(DEVICE_ID), "3400_A0"}},
        {{CONFIG_KEY(DEVICE_ID), "3400"}},
        {{CONFIG_KEY(DEVICE_ID), "3700"}},
        {{CONFIG_KEY(DEVICE_ID), "3800"}},
        {{CONFIG_KEY(DEVICE_ID), "3900"}},
        {{CONFIG_KEY(DEVICE_ID), "37XX"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, CompileForDifferentPlatformsTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                ::testing::ValuesIn(configs)),
                        CompileForDifferentPlatformsTests::getTestCaseName);
}  // namespace
