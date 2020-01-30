// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior_test_plugin_infer_request_config.hpp"

#include "vpu_test_data.hpp"

// double free detected
// [Track number: S#xxxxx]
INSTANTIATE_TEST_CASE_P(DISABLED_BehaviorTest, BehaviorPluginTestInferRequestConfig,
    ValuesIn(BehTestParams::concat(withCorrectConfValues, withCorrectConfValuesNetworkOnly)), getConfigTestCaseName);

// [Track number: S#xxxxx]
INSTANTIATE_TEST_CASE_P(DISABLED_BehaviorTest, BehaviorPluginTestInferRequestConfigExclusiveAsync,
    ValuesIn(supportedValues), getConfigTestCaseName);
