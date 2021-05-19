// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "vpux/vpux_plugin_config.hpp"
#include "common/functions.h"

using CompileForDifferentPlatformsTests = BehaviorTestsUtils::BehaviorTestsBasic;
namespace {

TEST_P(CompileForDifferentPlatformsTests, CompilationForSpecificPlatform) {
#if defined(__arm__) || defined(__aarch64__)
    SKIP() << "Compilation only";
#endif
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::map<std::string, std::string>> configs = {
        {{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3400_A0)}},
        {{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3400)}},
        {{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3700)}},
        {{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3800)}},
        {{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3900)}},
        {{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3720)}}};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, CompileForDifferentPlatformsTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                ::testing::ValuesIn(configs)),
                        CompileForDifferentPlatformsTests::getTestCaseName);
}  // namespace
