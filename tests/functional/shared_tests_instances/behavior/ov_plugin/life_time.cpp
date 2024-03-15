// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "overload/ov_plugin/life_time.hpp"
#include "vpu_test_tool.hpp"

using namespace ov::test::behavior;

namespace {

static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device +
           "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU) + "_";
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTest, ::testing::Values(ov::test::utils::DEVICE_NPU),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::DEVICE_NPU), getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestVpux, ::testing::Values(ov::test::utils::DEVICE_NPU),
                         getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetworkVpux,
                         ::testing::Values(ov::test::utils::DEVICE_NPU), getTestCaseName);

}  // namespace
