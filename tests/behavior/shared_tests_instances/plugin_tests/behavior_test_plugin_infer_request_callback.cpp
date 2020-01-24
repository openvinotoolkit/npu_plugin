// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_infer_request_callback.hpp"

#include "vpu_test_data.hpp"

INSTANTIATE_TEST_CASE_P(
    DISABLED_BehaviorTest, BehaviorPluginTestInferRequestCallback, ValuesIn(requestsSupportedValues), getTestCaseName);
