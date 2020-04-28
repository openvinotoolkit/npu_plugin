//
// Copyright 2019-2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <executable_network.h>
#include <hddl2_helpers/helper_remote_blob.h>
#include <hddl2_helpers/helper_remote_memory.h>
#include <hddl2_helpers/helper_workload_context.h>
#include <helper_remote_context.h>
#include <models/precompiled_resnet.h>

#include <chrono>
#include <ie_core.hpp>
#include <thread>

#include "creators/creator_blob_nv12.h"
#include "ie_metric_helpers.hpp"
#include "models/model_pooling.h"

namespace IE = InferenceEngine;

// TODO Use ImportNetwork tests as base
class InferRequest_Tests : public CoreAPI_Tests {
public:
    modelBlobInfo blobInfo = PrecompiledResNet_Helper::resnet50_dpu;

protected:
    void SetUp() override;
};

void InferRequest_Tests::SetUp() {
    // TODO Workaround [Track number: S#28523]
    std::this_thread::sleep_for(std::chrono::seconds(2));

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

//------------------------------------------------------------------------------
TEST_F(InferRequest_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

TEST_F(InferRequest_Tests, CanCallInference) {
    inferRequest = executableNetwork.CreateInferRequest();

    ASSERT_NO_THROW(inferRequest.Infer());
}

//------------------------------------------------------------------------------
// TODO [Add tests] Set NV12Blob preprocessing information inside cnnNetwork
using InferRequest_SetBlob = InferRequest_Tests;
TEST_F(InferRequest_SetBlob, CanSetInputBlob) {
    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(inputTensorDesc);
    blob->allocate();

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, blob));
}

TEST_F(InferRequest_SetBlob, CanSetInput_RemoteBlob) {
    WorkloadContext_Helper workloadContextHelper;
    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    WorkloadID id = workloadContextHelper.getWorkloadId();
    InferenceEngine::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(id);
    IE::RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    ASSERT_NE(remoteContext, nullptr);

    RemoteMemory_Helper remoteMemory;
    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    RemoteMemoryFd memoryFd = remoteMemory.allocateRemoteMemory(id, inputTensorDesc);
    auto blobParams = RemoteBlob_Helper::wrapRemoteFdToMap(memoryFd);
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContext->CreateBlob(inputInfoPtr->getTensorDesc(), blobParams);
    ASSERT_NE(nullptr, remoteBlobPtr);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, remoteBlobPtr));
}

// [Track number: S#30141]
TEST_F(InferRequest_SetBlob, CanSetInput_NV12Blob_WithPreprocessData) {
    inferRequest = executableNetwork.CreateInferRequest();
    ASSERT_EQ(executableNetwork.GetInputsInfo().size(), 1);

    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    auto nv12Blob = NV12Blob_Creator::createBlob(inputInfoPtr->getTensorDesc());
    auto preProcess = IE::PreProcessInfo();
    preProcess.setResizeAlgorithm(IE::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcess.setColorFormat(IE::ColorFormat::NV12);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, nv12Blob, preProcess));
}

//------------------------------------------------------------------------------
using InferRequest_GetBlob = InferRequest_Tests;
TEST_F(InferRequest_GetBlob, CanGetOutputBlobAfterInference) {
    inferRequest = executableNetwork.CreateInferRequest();

    inferRequest.Infer();

    std::string outputName = executableNetwork.GetOutputsInfo().begin()->first;
    InferenceEngine::Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(outputName));
}

TEST_F(InferRequest_GetBlob, GetBlobWillContainsSameDataAsSetBlob_WithRemoteMemory) {
    WorkloadContext_Helper workloadContextHelper;
    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    WorkloadID id = workloadContextHelper.getWorkloadId();
    InferenceEngine::ParamMap contextParams = Remote_Context_Helper::wrapWorkloadIdToMap(id);
    IE::RemoteContext::Ptr remoteContext = ie.CreateContext(pluginName, contextParams);
    ASSERT_NE(remoteContext, nullptr);

    RemoteMemory_Helper remoteMemory;
    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    RemoteMemoryFd memoryFd = remoteMemory.allocateRemoteMemory(id, inputTensorDesc);
    auto blobParams = RemoteBlob_Helper::wrapRemoteFdToMap(memoryFd);
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContext->CreateBlob(inputInfoPtr->getTensorDesc(), blobParams);
    ASSERT_NE(nullptr, remoteBlobPtr);

    const std::string inputData = "Test data";
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto rBlobData = lockedMemory.as<char*>();
        memcpy(rBlobData, inputData.data(), inputData.size());
    }

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, remoteBlobPtr));

    std::string resultData;
    {
        IE::Blob::Ptr inputBlob = inferRequest.GetBlob(inputName);
        auto inputBlobData = inputBlob->buffer().as<char*>();
        resultData = std::string(inputBlobData);
    }

    ASSERT_EQ(inputData, resultData);
}

//------------------------------------------------------------------------------
using InferRequestCreation_Tests = CoreAPI_Tests;
// [Track number: S#30141]
TEST_F(InferRequestCreation_Tests, DISABLED_CanCompileButCanNotCreateRequestWithoutDaemon) {
    unsetenv("KMB_INSTALL_DIR");
    ModelPooling_Helper modelPoolingHelper;
    auto cnnNetwork = modelPoolingHelper.network;

    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(cnnNetwork, pluginName));
    ASSERT_ANY_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

//------------------------------------------------------------------------------
class Inference_onSpecificDevice : public InferRequest_Tests {
public:
    int amountOfDevices = 0;

protected:
    void SetUp() override;
};

void Inference_onSpecificDevice::SetUp() {
    std::vector<HddlUnite::Device> devices;
    getAvailableDevices(devices);
    amountOfDevices = devices.size();
}

TEST_F(Inference_onSpecificDevice, CanInferOnSpecificDeviceFromPluginMetrics) {
    std::vector<std::string> availableDevices = ie.GetMetric(pluginName, METRIC_KEY(AVAILABLE_DEVICES));
    ASSERT_TRUE(!availableDevices.empty());

    const std::string device_name = pluginName + "." + availableDevices[0];
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, device_name));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_F(Inference_onSpecificDevice, CanInferOnSpecificDeviceFromGetAllDevices) {
    if (amountOfDevices <= 1) {
        GTEST_SKIP() << "Not enough devices for test";
    }
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    ASSERT_TRUE(!availableDevices.empty());

    std::vector<std::string> HDDL2Devices;
    std::copy_if(availableDevices.begin(), availableDevices.end(), std::back_inserter(HDDL2Devices),
        [this](const std::string& deviceName) {
            return deviceName.find(pluginName) != std::string::npos;
        });

    ASSERT_TRUE(!HDDL2Devices.empty());

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, HDDL2Devices[0]));
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inferRequest.Infer());
}
