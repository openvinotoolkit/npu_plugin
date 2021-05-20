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

        // [Track number: E#12898]
        ".*MLIR.*",
    };
}
