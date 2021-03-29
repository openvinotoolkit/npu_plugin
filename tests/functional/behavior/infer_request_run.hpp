// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
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
#include <cpp_interfaces/exception2status.hpp>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestRunTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferRequestRunTests, AllocatorCanDisposeBlobWhenOnlyInferRequestIsInScope) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        InferenceEngine::InferRequest req;
        InferenceEngine::Blob::Ptr OutputBlob;
        {
            InferenceEngine::SizeVector inputShape = {1, 3, 4, 3};
            InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;
            size_t axis = 1;

            const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

            const auto paramOuts = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

            const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);

            const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};

            function = std::make_shared<ngraph::Function>(results, params, "softMax");
            // Create CNNNetwork from ngraph::Function
            InferenceEngine::CNNNetwork cnnNet(function);
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
}  // namespace BehaviorTestsDefinitions
