// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "behavior/infer_request/perf_counters.hpp"

#include "overload/overload_test_utils_vpux.hpp"

namespace BehaviorTestsDefinitions {
class InferRequestPerfCountersTestVpux : public InferRequestPerfCountersTest {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDeviceVpux();
        cnnNet = InferenceEngine::CNNNetwork(function);
        configuration.insert(
                {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES});
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    }
};

TEST_P(InferRequestPerfCountersTestVpux, NotEmptyAfterAsyncInfer) {
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = req.GetPerformanceCounts());
    ASSERT_NE(perfMap.size(), 0);
}

TEST_P(InferRequestPerfCountersTestVpux, NotEmptyAfterSyncInfer) {
    InferenceEngine::CNNNetwork cnnNet(function);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_FATAL_FAILURE(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    ASSERT_NO_THROW(perfMap = req.GetPerformanceCounts());
    ASSERT_NE(perfMap.size(), 0);
}

}  // namespace BehaviorTestsDefinitions
