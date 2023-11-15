//
// Copyright (C) 2018-2020 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/plugin/version.hpp"
#include "hetero/hetero_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, VersionTest, ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                         VersionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, VersionTest, ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                         VersionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, VersionTest, ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                         VersionTest::getTestCaseName);

}  // namespace
