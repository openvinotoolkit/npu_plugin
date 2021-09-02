// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/executable_network/exec_graph_info.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {};

// double free detected
// [Track number: S#27337]
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                        ::testing::Combine(
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                         ExecutableNetworkBaseTest::getTestCaseName);
}  // namespace
