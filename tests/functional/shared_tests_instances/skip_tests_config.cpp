// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*ActivationLayerTest.*",
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

        // [Track number: S#42747]
        ".*KmbFakeQuantizeLayerTest\\.FakeQuantizeCheck.*",

        // [Track number: S#42749]
        ".*KmbSqueezeUnsqueezeLayerTest\\.BasicTest.*",

        // [Track number: S#43484]
        ".*KmbMaxMinLayerTest.*",

        // Need to create openvino releases/2020/vpux/2021/2 branch with #3350 pull request to fix test names
        ".*IEClassBasicTestP_smoke/IEClassBasicTestP.*",
        ".*IEClassGetMetricTest_nightly/IEClassGetMetricTest.*",

        // "Unable to deduce parameter 'align' for 'Interp' layer. Name is: 'interpolation', parameter is: 'align_corners'"
        ".*KmbInterpLayerTests.EqualWithCPU/1",
        ".*KmbInterpLayerTests.EqualWithCPU/3",
        ".*KmbInterpLayerTests.EqualWithCPU/5",

        // Windows Only: "LpScheduler - RuntimeError: input is not a DAG"
        ".*precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32",
    };
}
