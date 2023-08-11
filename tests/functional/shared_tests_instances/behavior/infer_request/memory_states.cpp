// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/memory_states.hpp"
#include <openvino/runtime/device_id_parser.hpp>
#include "common/functions.h"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

std::string getDeviceName() {
    auto* env_val = std::getenv("IE_KMB_TESTS_DEVICE_NAME");
    return (env_val != nullptr) ? env_val : "VPUX.3720";
}

const ov::DeviceIDParser parser = ov::DeviceIDParser(getDeviceName());

std::string getDeviceNameTestCase() {
    return parser.get_device_name().substr(0, parser.get_device_name().size() - 1) + parser.get_device_id();
}

static std::string getTestCaseName(testing::TestParamInfo<memoryStateParams> obj) {
    InferenceEngine::CNNNetwork network;
    std::vector<std::string> memoryStates;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(network, memoryStates, targetDevice, configuration) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << getDeviceNameTestCase() << "_";
    return result.str();
}

std::vector<memoryStateParams> memoryStateTestCases = {memoryStateParams(
        InferRequestVariableStateTest::getNetwork(), {"c_1-3", "r_1-3"}, CommonTestUtils::DEVICE_KEEMBAY, {})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableStateBasic, InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases), getTestCaseName);
