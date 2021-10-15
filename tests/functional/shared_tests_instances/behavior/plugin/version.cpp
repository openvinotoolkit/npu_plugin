// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero/hetero_plugin_config.hpp"
#include "behavior/plugin/version.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, VersionTest,
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                        VersionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, VersionTest,
                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                        VersionTest::getTestCaseName);


}  // namespace
