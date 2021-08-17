// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <base/behavior_test_utils.hpp>
#include <ie_core.hpp>
#include <thread>
#include "common/functions.h"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "vpux_private_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ie_extension.h"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestRunTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestRunTests, AllocatorCanDisposeBlobWhenOnlyInferRequestIsInScope) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        InferenceEngine::InferRequest req;
        InferenceEngine::Blob::Ptr outputBlob;
        {
            InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
#ifdef __aarc64__
            // Load CNNNetwork to target plugins
            auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
            // Create InferRequest
            ASSERT_NO_THROW(req = execNet.CreateInferRequest());
            InferenceEngine::Blob::Ptr inputBlob =
                    FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
            ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
            req.Infer();
            ASSERT_NO_THROW(outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
            ASSERT_EQ(outputBlob->cbuffer().as<const float*>()[0], outputBlob->cbuffer().as<const float*>()[0]);
#endif
            ie.reset();
            PluginCache::get().reset();
        }
#ifdef __aarc64__
        std::cout << "Final access to blob begin, should not cause segmentation fault" << std::endl;
        ASSERT_EQ(outputBlob->cbuffer().as<const float*>()[0], outputBlob->cbuffer().as<const float*>()[0]);
        std::cout << "Final access to blob ended, should not cause segmentation fault" << std::endl;
#endif
    }
    std::cout << "Plugin should be unloaded from memory at this point" << std::endl;
}

using InferRequestRunMultipleExecutorStreamsTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestRunMultipleExecutorStreamsTests, RunFewSyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();

    // Load CNNNetwork to target plugins
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequests
    const int inferReqNumber = 10;
    std::array<InferenceEngine::InferRequest, inferReqNumber> inferReqs;
    std::array<std::thread, inferReqNumber> inferReqsThreads;
    for (int i = 0; i < inferReqNumber; ++i) {
        ASSERT_NO_THROW(inferReqs[i] = execNet.CreateInferRequest());

        InferenceEngine::Blob::Ptr inputBlob =
                FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        ASSERT_NO_THROW(inferReqs[i].SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        InferenceEngine::InferRequest &infReq = inferReqs[i];
        inferReqsThreads[i] = std::thread([&infReq]() -> void {
	    ASSERT_NO_THROW(infReq.Infer());
        });
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqsThreads[i].join();
        ASSERT_NO_THROW(inferReqs[i].GetBlob(cnnNet.getOutputsInfo().begin()->first));
    }
}

TEST_P(InferRequestRunMultipleExecutorStreamsTests, RunFewAsyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();

    // Load CNNNetwork to target plugins
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequests
    const int inferReqNumber = 10;
    std::array<InferenceEngine::InferRequest, inferReqNumber> inferReqs;
    for (int i = 0; i < inferReqNumber; ++i) {
        ASSERT_NO_THROW(inferReqs[i] = execNet.CreateInferRequest());

        InferenceEngine::Blob::Ptr inputBlob =
                FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        ASSERT_NO_THROW(inferReqs[i].SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
        inferReqs[i].StartAsync();
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqs[i].Wait(InferenceEngine::InferRequest::RESULT_READY);
        ASSERT_NO_THROW(inferReqs[i].GetBlob(cnnNet.getOutputsInfo().begin()->first));
    }
}

}  // namespace BehaviorTestsDefinitions
