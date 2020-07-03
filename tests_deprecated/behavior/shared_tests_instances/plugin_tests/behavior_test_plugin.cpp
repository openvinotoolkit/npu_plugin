// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

#include "behavior_test_plugins.hpp"
#include "vpu_test_data.hpp"

// [Track number: S#27334]
INSTANTIATE_TEST_CASE_P(DISABLED_BehaviorTest, BehaviorPluginTest, ValuesIn(supportedValues), getTestCaseName);

// [Track number: S#27334]
INSTANTIATE_TEST_CASE_P(
    DISABLED_BehaviorTest, BehaviorPluginTestInput, ValuesIn(allInputSupportedValues), getTestCaseName);

// [Track number: S#27334]
INSTANTIATE_TEST_CASE_P(
    DISABLED_BehaviorTest, BehaviorPluginTestOutput, ValuesIn(allOutputSupportedValues), getOutputTestCaseName);
