// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO Tests failed due to starting infer on IA side
        ".*CorrectConfigAPITests.*",

        // double free detected
        // [Track number: S#27343]
        ".*InferConfigInTests\\.CanInferWithConfig.*",
        ".*InferConfigTests\\.withoutExclusiveAsyncRequests.*",
        ".*InferConfigTests\\.canSetExclusiveAsyncRequests.*",

        // [Track number: S#27334]
        ".*BehaviorTests.*",
        ".*BehaviorTestInput.*",
        ".*BehaviorTestOutput.*",

        // [Track number: S#42749]
        ".*KmbSqueezeUnsqueezeLayerTest\\.BasicTest.*",

        // [Track number: S#43484]
        ".*KmbMaxMinLayerTest.*",
    };
}
