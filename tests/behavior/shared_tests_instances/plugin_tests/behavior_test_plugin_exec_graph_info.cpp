// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_exec_graph_info.hpp"

#include "vpu_test_data.hpp"

// double free detected
// [Track number: S#27337]
INSTANTIATE_TEST_CASE_P(
    DISABLED_BehaviorTest, BehaviorPluginTestExecGraphInfo, ValuesIn(supportedValues), getTestCaseName);
