//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/ov_infer_request/callback.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "overload/ov_infer_request/callback.hpp"
#include "vpu_test_tool.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_NPU}}};

const std::vector<ov::AnyMap> autoConfigs = {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_NPU}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestCallbackTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCallbackTestsVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCallbackTestsVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestCallbackTestsVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

}  // namespace
