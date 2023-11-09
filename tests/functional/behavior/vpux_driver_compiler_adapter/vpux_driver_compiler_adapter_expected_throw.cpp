//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_driver_compiler_adapter_expected_throw.hpp"
#include "vpux/al/config/common.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{{ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, VpuDriverCompilerAdapterExpectedThrow,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         VpuDriverCompilerAdapterExpectedThrow::getTestCaseName);
