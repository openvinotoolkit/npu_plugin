//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/ov_infer_request/wait.hpp"
#include "ie/ie_plugin_config.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_KEEMBAY}}};

const std::vector<ov::AnyMap> autoConfigs = {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_KEEMBAY}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         OVInferRequestWaitTests::getTestCaseName);

}  // namespace
