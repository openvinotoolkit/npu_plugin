//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/infer_request_run.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configsInferRequestRunTests = {{{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, InferRequestRunTests,
                         ::testing::Combine(::testing::Values(LayerTestsUtils::getDeviceName()),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);
