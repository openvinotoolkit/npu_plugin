// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include "ie_extension.h"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "common/functions.h"
#include "vpux_private_config.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestRunTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestRunTests, AllocatorCanDisposeBlobWhenOnlyInferRequestIsInScope) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        InferenceEngine::InferRequest req;
        InferenceEngine::Blob::Ptr OutputBlob;
        {
            InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
#ifdef __aarc64__
            // Load CNNNetwork to target plugins
            auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
            // Create InferRequest
            ASSERT_NO_THROW(req = execNet.CreateInferRequest());
            InferenceEngine::Blob::Ptr InputBlob =
                    FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
            ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, InputBlob));
            req.Infer();
            ASSERT_NO_THROW(OutputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
            ASSERT_EQ(OutputBlob->cbuffer().as<const float*>()[0], OutputBlob->cbuffer().as<const float*>()[0]);
#endif
            ie.reset();
            PluginCache::get().reset();
        }
#ifdef __aarc64__
        std::cout << "Final access to blob begin, should not cause segmentation fault" << std::endl;
        ASSERT_EQ(OutputBlob->cbuffer().as<const float*>()[0], OutputBlob->cbuffer().as<const float*>()[0]);
        std::cout << "Final access to blob ended, should not cause segmentation fault" << std::endl;
#endif
    }
    std::cout << "Plugin should be unloaded from memory at this point" << std::endl;
}

using InferRequestRunMultipleExecutorStreamsTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestRunMultipleExecutorStreamsTests, RunFewInfers) {
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

        InferenceEngine::Blob::Ptr InputBlob =
                FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        ASSERT_NO_THROW(inferReqs[i].SetBlob(cnnNet.getInputsInfo().begin()->first, InputBlob));
        inferReqs[i].StartAsync();
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqs[i].Wait(InferenceEngine::InferRequest::RESULT_READY);
        ASSERT_NO_THROW(inferReqs[i].GetBlob(cnnNet.getOutputsInfo().begin()->first));
    }
}

}  // namespace BehaviorTestsDefinitions
