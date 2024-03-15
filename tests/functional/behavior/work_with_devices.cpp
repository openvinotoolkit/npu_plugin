// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/work_with_devices.hpp"
#include "common/utils.hpp"

namespace {

const std::vector<ov::AnyMap> configs = {
        {{ov::log::level(ov::log::Level::DEBUG)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, NPUTestCompiledModel,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         NPUTestCompiledModel::getTestCaseName);

}  // namespace
