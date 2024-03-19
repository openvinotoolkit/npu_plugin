// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/infer_request/wait.hpp"
#include "common/utils.hpp"
#include "ie_plugin_config.hpp"
#include "overload/infer_request/wait.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestWaitTests::getTestCaseName);
// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestWaitTestsVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestWaitTestsVpux::getTestCaseName);
}  // namespace
