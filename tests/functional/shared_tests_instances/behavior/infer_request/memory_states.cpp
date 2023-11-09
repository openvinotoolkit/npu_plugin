//
// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/infer_request/memory_states.hpp"
#include "common/functions.h"
#include "common_test_utils/test_constants.hpp"
#include "vpu_test_env_cfg.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

static std::string getTestCaseName(testing::TestParamInfo<memoryStateParams> obj) {
    InferenceEngine::CNNNetwork network;
    std::vector<std::string> memoryStates;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(network, memoryStates, targetDevice, configuration) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << LayerTestsUtils::getDeviceNameTestCase(targetDevice) << "_";
    return result.str();
}

std::vector<memoryStateParams> memoryStateTestCases = {memoryStateParams(
        InferRequestVariableStateTest::getNetwork(), {"c_1-3", "r_1-3"}, CommonTestUtils::DEVICE_KEEMBAY, {})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableStateBasic, InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases), getTestCaseName);
