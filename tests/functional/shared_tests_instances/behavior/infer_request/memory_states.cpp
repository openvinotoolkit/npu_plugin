// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/infer_request/memory_states.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

static std::string getTestCaseName(testing::TestParamInfo<memoryStateParams> obj) {
    InferenceEngine::CNNNetwork network;
    std::vector<std::string> memoryStates;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(network, memoryStates, targetDevice, configuration) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU) << "_";
    return result.str();
}

std::vector<memoryStateParams> memoryStateTestCases = {memoryStateParams(
        InferRequestVariableStateTest::getNetwork(), {"c_1-3", "r_1-3"}, ov::test::utils::DEVICE_NPU, {})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableStateBasic, InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases), getTestCaseName);
