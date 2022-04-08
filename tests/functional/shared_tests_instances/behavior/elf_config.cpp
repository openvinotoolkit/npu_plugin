// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <base/behavior_test_utils.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "vpux_private_config.hpp"

using ElfConfigTests = BehaviorTestsUtils::BehaviorTestsBasic;
namespace {

TEST_P(ElfConfigTests, CompilationWithSpecificConfig) {
    if (getBackendName(*ie) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, target_device, configuration));
    }
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {
        {{VPUX_CONFIG_KEY(PLATFORM), "VPU3720"}, {VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND), "NO"}}};

INSTANTIATE_TEST_SUITE_P(smoke_ELF, ElfConfigTests,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         ElfConfigTests::getTestCaseName);
}  // namespace
