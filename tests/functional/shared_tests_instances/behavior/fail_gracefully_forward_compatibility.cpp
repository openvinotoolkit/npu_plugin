//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/fail_gracefully_forward_compatibility.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {
        {{"NPU_COMPILER_TYPE", "DRIVER"}},
        {{"NPU_COMPILER_TYPE", "MLIR"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, FailGracefullyTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         FailGracefullyTest::getTestCaseName);
