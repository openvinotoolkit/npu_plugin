// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_executable_network/exec_network_base.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVExecutableNetworkBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);
}  // namespace
