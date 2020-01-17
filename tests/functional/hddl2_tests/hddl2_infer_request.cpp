//
// Copyright 2019 Intel Corporation.
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

#include <Inference.h>
#include <hddl2_helpers/helper_model_loader.h>
#include <hddl2_helpers/helper_precompiled_resnet.h>

#include <ie_core.hpp>

#include "gtest/gtest.h"
#include "hddl2_core_api.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_workload_context.h"
#include "helper_remote_context.h"
#include "ie_blob.h"
#include "parametric_executable_network.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_InferRequest_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_InferRequest_Tests : public Executable_Network_Parametric {
public:
    InferenceEngine::InferRequest inferRequest;
};

//------------------------------------------------------------------------------
//      class HDDL2_InferRequest_Tests Initiation - create
//------------------------------------------------------------------------------
TEST_P(HDDL2_InferRequest_Tests, CanCreateInferRequest) {
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

//------------------------------------------------------------------------------
//      class HDDL2_InferRequest_Tests Initiation - Infer
//------------------------------------------------------------------------------
TEST_P(HDDL2_InferRequest_Tests, CanCallInference) {
    // TODO Enable after LoadNetwork implementation
    if (GetParam() == LoadNetwork) {
        SKIP() << "LoadNetwork not fully implemented";
    }
    inferRequest = executableNetwork.CreateInferRequest();

    ASSERT_NO_THROW(inferRequest.Infer());
}

//------------------------------------------------------------------------------
//      class HDDL2_InferRequest_Tests Initiation - SetBlob
//------------------------------------------------------------------------------
TEST_P(HDDL2_InferRequest_Tests, CanSetInputBlob) {
    // TODO Enable after LoadNetwork implementation
    if (GetParam() == LoadNetwork) {
        SKIP() << "Incorrect blob type. Doesn not have correct parsed parameters";
    }

    inferRequest = executableNetwork.CreateInferRequest();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    IE::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    IE::TensorDesc inputTensorDesc = inputInfoPtr->getTensorDesc();
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(inputTensorDesc);
    blob->allocate();

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, blob));
}

TEST_P(HDDL2_InferRequest_Tests, CanSetInputBlob_WithRemoteBlob) {
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

//------------------------------------------------------------------------------
//      class HDDL2_InferRequest_Tests Initiation - GetBlob
//------------------------------------------------------------------------------
TEST_P(HDDL2_InferRequest_Tests, CanGetOutputBlobAfterInference) {
    // TODO Enable after LoadNetwork implementation
    if (GetParam() == LoadNetwork) {
        SKIP() << "LoadNetwork not fully implemented";
    }

    inferRequest = executableNetwork.CreateInferRequest();

    inferRequest.Infer();

    std::string outputName = executableNetwork.GetOutputsInfo().begin()->first;
    InferenceEngine::Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(outputName));
}

TEST_P(HDDL2_InferRequest_Tests, GetBlobWillContainsSameDataAsSetBlob_WithRemoteMemory) {
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
    // TODO Any other way to unlock memory?
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
//      class HDDL2_InferRequest_Tests Test case Initiations
//------------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(ExecNetworkFrom, HDDL2_InferRequest_Tests, ::testing::ValuesIn(memoryOwners),
    Executable_Network_Parametric::PrintToStringParamName());
