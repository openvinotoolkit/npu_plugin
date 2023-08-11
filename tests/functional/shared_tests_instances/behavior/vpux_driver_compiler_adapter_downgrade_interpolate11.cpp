// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux_driver_compiler_adapter_downgrade_interpolate11.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {
        {{"VPUX_COMPILER_TYPE", "DRIVER"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, VpuxDriverCompilerAdapterDowngradeInterpolate11Test,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         VpuxDriverCompilerAdapterDowngradeInterpolate11Test::getTestCaseName);
