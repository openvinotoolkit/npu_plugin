// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <base/behavior_test_utils.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>


namespace {
const auto expectedDeviceName = std::string{"VPUX"};
}

TEST(smoke_InterfaceTests, TestEngineClassGetMetric) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::Core ie;

        const std::vector<std::string> supportedMetrics =
                ie.GetMetric(expectedDeviceName, METRIC_KEY(SUPPORTED_METRICS));
        auto supportedConfigKeysFound = false;
        for (const auto& metricName : supportedMetrics) {
            if (metricName == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                ASSERT_FALSE(supportedConfigKeysFound);  // may be we should make more strict test for duplicates of key
                supportedConfigKeysFound = true;
            }
            ASSERT_FALSE(ie.GetMetric(expectedDeviceName, metricName).empty());
        }
        ASSERT_TRUE(supportedConfigKeysFound);  // plus implicit check for !supportedMetrics.empty()

        ASSERT_THROW(ie.GetMetric(expectedDeviceName, "THISMETRICNOTEXIST"),
                     InferenceEngine::details::InferenceEngineException);
    }
}

TEST(smoke_InterfaceTests, TestEngineClassGetConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::Core ie;

        const std::vector<std::string> supportedConfigKeys =
                ie.GetMetric(expectedDeviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        ASSERT_FALSE(supportedConfigKeys.empty());
        for (const auto& configKey : supportedConfigKeys) {
            ASSERT_FALSE(ie.GetConfig(expectedDeviceName, configKey).empty());
        }

        ASSERT_THROW(ie.GetConfig(expectedDeviceName, "THISKEYNOTEXIST"),
                     InferenceEngine::details::InferenceEngineException);
    }
}
