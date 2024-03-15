// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/plugin/version.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "hetero/hetero_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string targetDevice;
    std::map<std::string, std::string> config;
    targetDevice = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
    std::ostringstream result;
    result << "targetDevice=" << targetDevice;
    result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
           << "_";
    return result.str();
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, VersionTest, ::testing::Values(ov::test::utils::DEVICE_NPU),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, VersionTest, ::testing::Values(ov::test::utils::DEVICE_AUTO),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, VersionTest, ::testing::Values(ov::test::utils::DEVICE_HETERO),
                         getTestCaseName);

}  // namespace
