// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "functional_test_utils/plugin_cache.hpp"
#include "common_test_utils/common_utils.hpp"
#include "kmb_layer_test.hpp"
#include "common/functions.h"


const std::vector<std::string> DISABLED_TESTS = {
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

    // TODO Add safe Softplus support
    ".*ActivationLayerTest.*SoftPlus.*",

    // TODO: Issue: 63469
    ".*KmbConversionLayerTest.*ConvertLike.*"
};

const std::vector<std::string> TESTS_TO_SKIP_IF_NO_DEVICE_AVAILABLE = {
    // Cannot run InferRequest tests without a device to infer to
    ".*BehaviorTests.InferRequest*.*",
};

const bool IS_DEVICE_AVAILABLE = []() -> bool {
    const auto corePtr = PluginCache::get().ie();
    if (corePtr == nullptr) {
        return false;
    }
    const auto backendName = getBackendName(*corePtr);
    const auto noDevice = backendName.empty();
    if (noDevice) {
        std::cout << "backend is empty (no device)"
                  << " - some tests might be skipped" << std::endl;
    }
    return !noDevice;
}();

const std::vector<std::string> ALL_DISABLED_TESTS_PATTERNS = []() {
    auto allDisabledTestPatterns = DISABLED_TESTS;
    if (!IS_DEVICE_AVAILABLE) {
        allDisabledTestPatterns.insert(
            allDisabledTestPatterns.end(),
            TESTS_TO_SKIP_IF_NO_DEVICE_AVAILABLE.cbegin(),
            TESTS_TO_SKIP_IF_NO_DEVICE_AVAILABLE.cend()
        );
    }
    return allDisabledTestPatterns;
}();

std::vector<std::string> disabledTestPatterns() {
    return ALL_DISABLED_TESTS_PATTERNS;
}
