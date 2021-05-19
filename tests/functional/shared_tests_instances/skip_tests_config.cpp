// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO Tests failed due to starting infer on IA side
        ".*CorrectConfigAPITests.*",

        // ARM CPU Plugin is not available on Yocto
        ".*IEClassLoadNetworkTest.*HETERO.*",
        ".*IEClassLoadNetworkTest.*MULTI.*",

        // Cannot detect vpu platform when it's not passed
        // Skip tests on Yocto which passes device without platform
        // [Track number: E#12774]
        ".*IEClassLoadNetworkTest.LoadNetworkWithDeviceIDNoThrow.*",
        ".*IEClassLoadNetworkTest.LoadNetworkWithBigDeviceIDThrows.*",
        ".*IEClassLoadNetworkTest.LoadNetworkWithInvalidDeviceIDThrows.*",

        // double free detected
        // [Track number: S#27343]
        ".*InferConfigInTests\\.CanInferWithConfig.*",
        ".*InferConfigTests\\.withoutExclusiveAsyncRequests.*",
        ".*InferConfigTests\\.canSetExclusiveAsyncRequests.*",

        // [Track number: S#27334]
        ".*BehaviorTests.*",
        ".*BehaviorTestInput.*",
        ".*BehaviorTestOutput.*",

        // [Track number: s#47412]
        ".*IEClassGetConfigTest_ThrowUnsupported\\.GetConfigThrow.*",

        // TODO Add safe Softplus support
        ".*ActivationLayerTest.*SoftPlus.*"
    };
}
