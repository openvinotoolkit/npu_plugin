// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_config.hpp"

#include "vpu_test_data.hpp"

INSTANTIATE_TEST_CASE_P(DISABLED_BehaviorTest, BehaviorPluginCorrectConfigTest,
    ValuesIn(BehTestParams::concat(withCorrectConfValues, withCorrectConfValuesPluginOnly)), getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_BehaviorTest, BehaviorPluginIncorrectConfigTest, ValuesIn(withIncorrectConfValues), getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_BehaviorTest, BehaviorPluginIncorrectConfigTestInferRequestAPI,
    ValuesIn(supportedValues), getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    DISABLED_BehaviorTest, BehaviorPluginCorrectConfigTestInferRequestAPI, ValuesIn(supportedValues), getTestCaseName);
