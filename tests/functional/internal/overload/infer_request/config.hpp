// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "behavior/infer_request/config.hpp"
#include "common/utils.hpp"
#include "overload/overload_test_utils_vpux.hpp"

namespace BehaviorTestsDefinitions {

class InferRequestConfigTestVpux : public InferRequestConfigTest {
public:
    void SetUp() override {
        std::tie(streamExecutorNumber, target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        // Create CNNNetwork from ngrpah::Function
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDeviceVpux();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }
};

TEST_P(InferRequestConfigTestVpux, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, InferenceEngine::executorManager()->getExecutorsNumber());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, InferenceEngine::executorManager()->getExecutorsNumber());
    }
}

TEST_P(InferRequestConfigTestVpux, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, InferenceEngine::executorManager()->getExecutorsNumber());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, InferenceEngine::executorManager()->getExecutorsNumber());
    }
}

TEST_P(InferRequestConfigTestVpux, ReusableCPUStreamsExecutor) {
    ASSERT_EQ(0u, InferenceEngine::executorManager()->getExecutorsNumber());
    ASSERT_EQ(0u, InferenceEngine::executorManager()->getIdleCPUStreamsExecutorsNumber());

    {
        // Load config
        std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)}};
        config.insert(configuration.begin(), configuration.end());
        if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
            ASSERT_NO_THROW(ie->SetConfig(config, target_device));
        }
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, target_device, config);
        execNet.CreateInferRequest();
        if (target_device == ov::test::utils::DEVICE_NPU) {
            ASSERT_EQ(1u, InferenceEngine::executorManager()->getExecutorsNumber());
            ASSERT_EQ(0u, InferenceEngine::executorManager()->getIdleCPUStreamsExecutorsNumber());
        } else if ((target_device == ov::test::utils::DEVICE_AUTO) ||
                   (target_device == ov::test::utils::DEVICE_MULTI)) {
        } else {
            ASSERT_EQ(0u, InferenceEngine::executorManager()->getExecutorsNumber());
            ASSERT_GE(2u, InferenceEngine::executorManager()->getIdleCPUStreamsExecutorsNumber());
        }
    }
}
}  // namespace BehaviorTestsDefinitions
