// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/plugin/life_time.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::vector<int>> orders = {
        // 0 - plugin
        // 1 - executable_network
        // 2 - infer_request
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

static std::string getTestCaseName(testing::TestParamInfo<HoldersParams> obj) {
    std::string target_device;
    std::vector<int> order;
    std::tie(target_device, order) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
           << "_";
    if (!order.empty()) {
        std::string objects[] = {"core", "exec.net", "request", "state"};
        for (auto& Item : order) {
            result << objects[Item] << "_";
        }
    }
    return result.str();
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, HoldersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(orders)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, HoldersTestImportNetwork,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(orders)),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, HoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::DEVICE_NPU), HoldersTestOnImportedNetwork::getTestCaseName);

}  // namespace
