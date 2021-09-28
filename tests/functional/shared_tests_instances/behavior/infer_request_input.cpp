// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request_input.hpp"
#include "ie_plugin_config.hpp"
#include "kmb_layer_test.hpp"
#include "common/functions.h"

using namespace LayerTestsUtils;

namespace BehaviorTestsDefinitions {

class VpuxInferRequestInputTests :
        public InferRequestInputTests, 
        virtual public KmbLayerTestsCommon {
                
    void SetUp() override {
        std::tie(KmbLayerTestsCommon::inPrc,
                 KmbLayerTestsCommon::targetDevice,
                 KmbLayerTestsCommon::configuration) = this->GetParam();

        KmbLayerTestsCommon::outPrc = KmbLayerTestsCommon::inPrc;
        KmbLayerTestsCommon::function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void SkipBeforeLoad() override {
        const std::string backendName = getBackendName(*core);
        const auto noDevice = backendName.empty();
        if (noDevice) {
            throw LayerTestsUtils::KmbSkipTestException("backend is empty (no device)");
        }
    }

    void TearDown() override {
        KmbLayerTestsCommon::TearDown();
    }
};

TEST_P(VpuxInferRequestInputTests, canSetInputBlobForSyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        const auto  inputsInfo     = KmbLayerTestsCommon::cnnNetwork.getInputsInfo();
        const auto& blobName       = inputsInfo.begin()->first;
        const auto& blobTensorDesc = inputsInfo.begin()->second->getTensorDesc();

        // Set input blob
        InferenceEngine::Blob::Ptr inputBlob = FuncTestUtils::createAndFillBlob(blobTensorDesc);
        ASSERT_NO_THROW(KmbLayerTestsCommon::inferRequest.SetBlob(blobName, inputBlob));
    
        // Get input blob
        InferenceEngine::Blob::Ptr actualBlob;
        ASSERT_NO_THROW(actualBlob = KmbLayerTestsCommon::inferRequest.GetBlob(blobName));
    
        // Compare the blobs
        ASSERT_EQ(inputBlob, actualBlob);
    }
}

TEST_P(VpuxInferRequestInputTests, canInferWithSetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        // Get name and tensorDesc for input blob
        const auto  inputsInfo          = KmbLayerTestsCommon::cnnNetwork.getInputsInfo();
        const auto& inputBlobName       = inputsInfo.begin()->first;
        const auto& inputBlobTensorDesc = inputsInfo.begin()->second->getTensorDesc();
        // Set input blob
        InferenceEngine::Blob::Ptr inputBlob = FuncTestUtils::createAndFillBlob(inputBlobTensorDesc);
        KmbLayerTestsCommon::inferRequest.SetBlob(inputBlobName, inputBlob);

        // Get name and tensorDesc for output blob
        const auto  outputsInfo          = KmbLayerTestsCommon::cnnNetwork.getOutputsInfo();
        const auto& outputBlobName       = outputsInfo.begin()->first;
        const auto& outputBlobTensorDesc = outputsInfo.begin()->second->getTensorDesc();
        // Set output blob
        InferenceEngine::Blob::Ptr outputBlob = FuncTestUtils::createAndFillBlob(outputBlobTensorDesc);
        KmbLayerTestsCommon::inferRequest.SetBlob(outputBlobName, outputBlob);
        
        // Infer
        ASSERT_NO_THROW(KmbLayerTestsCommon::inferRequest.Infer());
    }
}

TEST_P(VpuxInferRequestInputTests, canGetInputBlob_deprecatedAPI) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        const auto  inputsInfo          = KmbLayerTestsCommon::cnnNetwork.getInputsInfo();
        const auto& inputBlobName       = inputsInfo.begin()->first;
        const auto& inputBlobTensorDesc = inputsInfo.begin()->second->getTensorDesc();
        const auto& inputBlobPrecision  = inputsInfo.begin()->second->getPrecision();
        
        InferenceEngine::Blob::Ptr actualBlob;
        ASSERT_NO_THROW(actualBlob = KmbLayerTestsCommon::inferRequest.GetBlob(inputBlobName));
        ASSERT_TRUE(actualBlob) << "Plugin didn't allocate input blobs";
        ASSERT_FALSE(actualBlob->buffer() == nullptr) << "Plugin didn't allocate input blobs";

        const auto& tensorDescription = actualBlob->getTensorDesc();
        const auto& dims = tensorDescription.getDims();
        ASSERT_TRUE(inputBlobTensorDesc.getDims() == dims)
            << "Input blob dimensions don't match network input";

        ASSERT_EQ(inputBlobPrecision, tensorDescription.getPrecision())
            << "Input blob precision doesn't match network input";
    }
}

TEST_P(VpuxInferRequestInputTests, getAfterSetInputDoNotChangeInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        const auto  inputsInfo     = KmbLayerTestsCommon::cnnNetwork.getInputsInfo();
        const auto& blobName       = inputsInfo.begin()->first;
        const auto& blobTensorDesc = inputsInfo.begin()->second->getTensorDesc();

        // Set blob
        InferenceEngine::Blob::Ptr inputBlob = FuncTestUtils::createAndFillBlob(blobTensorDesc);
        ASSERT_NO_THROW(KmbLayerTestsCommon::inferRequest.SetBlob(blobName, inputBlob));
    
        // Get blob
        InferenceEngine::Blob::Ptr actualBlob;
        ASSERT_NO_THROW(actualBlob = KmbLayerTestsCommon::inferRequest.GetBlob(blobName));

        // Compare the blobs   
        ASSERT_EQ(inputBlob.get(), actualBlob.get());
    }
}

TEST_P(VpuxInferRequestInputTests, canInferWithGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        // Get input blob
        const auto inputsInfo  = KmbLayerTestsCommon::cnnNetwork.getInputsInfo();
        const auto& inputBlobName = inputsInfo.begin()->first;
        InferenceEngine::Blob::Ptr inputBlob = KmbLayerTestsCommon::inferRequest.GetBlob(inputBlobName);

        // Get output blob
        const auto outputsInfo = KmbLayerTestsCommon::cnnNetwork.getOutputsInfo();
        const auto& outputBlobName = outputsInfo.begin()->first;
        InferenceEngine::Blob::Ptr outputBlob = KmbLayerTestsCommon::inferRequest.GetBlob(outputBlobName);

        ASSERT_NO_THROW(KmbLayerTestsCommon::inferRequest.Infer());
    }
}

TEST_P(VpuxInferRequestInputTests, canStartAsyncInferWithGetInOut) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        // Get input blob name
        const auto inputsInfo  = KmbLayerTestsCommon::cnnNetwork.getInputsInfo();
        const auto& inputBlobName = inputsInfo.begin()->first;

        // Get output blob name
        const auto outputsInfo = KmbLayerTestsCommon::cnnNetwork.getOutputsInfo();
        const auto& outputBlobName = outputsInfo.begin()->first;

        // Get input blob
        InferenceEngine::Blob::Ptr inputBlob = KmbLayerTestsCommon::inferRequest.GetBlob(inputBlobName);
        
        // Async Infer
        InferenceEngine::StatusCode sts;
        ASSERT_NO_THROW(KmbLayerTestsCommon::inferRequest.Infer());
        ASSERT_NO_THROW(KmbLayerTestsCommon::inferRequest.StartAsync());
        sts = KmbLayerTestsCommon::inferRequest.Wait(500);
        ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);

        // Get output blob
        InferenceEngine::Blob::Ptr outputBlob = KmbLayerTestsCommon::inferRequest.GetBlob(outputBlobName);
    }
}

}  // namespace BehaviorTestsDefinitions

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, VpuxInferRequestInputTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(configs)),
                        VpuxInferRequestInputTests::getTestCaseName);
}  // namespace
