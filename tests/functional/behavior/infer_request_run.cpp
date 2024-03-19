// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/infer_request_run.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configsInferRequestRunTests = {{{ov::log::level(ov::log::Level::INFO)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, InferRequestRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);
