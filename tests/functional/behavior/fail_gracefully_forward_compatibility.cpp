// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/fail_gracefully_forward_compatibility.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "vpux/al/config/common.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {
        {{ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER)}},
        {{ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::MLIR)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, FailGracefullyTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         FailGracefullyTest::getTestCaseName);
