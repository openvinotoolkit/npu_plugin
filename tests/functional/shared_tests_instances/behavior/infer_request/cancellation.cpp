// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/infer_request/cancellation.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ie_plugin_config.hpp"
#include "overload/infer_request/cancellation.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsMapTestName::getTestCaseName);
// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCancellationTestsVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestCancellationTestsVpux::getTestCaseName);
}  // namespace
