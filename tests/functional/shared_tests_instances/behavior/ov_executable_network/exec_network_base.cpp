// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/exec_network_base.hpp"
// #include "ie_plugin_config.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVExecutableNetworkBaseTest,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                 ::testing::ValuesIn(configs)),
                         OVExecutableNetworkBaseTest::getTestCaseName);
}  // namespace//
