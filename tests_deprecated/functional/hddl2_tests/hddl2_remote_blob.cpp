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

#include "core_api.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "vpux/vpux_plugin_params.hpp"
#include "helper_remote_context.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class HDDL2_Remote_Blob_Tests : public CoreAPI_Tests {
public:
    void SetUp() override;
    void TearDown() override;

    IE::RemoteContext::Ptr remoteContextPtr = nullptr;

    IE::TensorDesc tensorDesc;
    VpuxRemoteMemoryFD remoteMemoryFD;

    RemoteMemory_Helper remoteMemoryHelper;
    const size_t memoryToAllocate = 1024 * 1024 * 4;
private:
    Remote_Context_Helper _remoteContextHelper;
    TensorDescription_Helper _tensorDescriptionHelper;
};

void HDDL2_Remote_Blob_Tests::SetUp() {
    remoteContextPtr = _remoteContextHelper.remoteContextPtr;
    tensorDesc = _tensorDescriptionHelper.tensorDesc;

    remoteMemoryFD =
        remoteMemoryHelper.allocateRemoteMemory(_remoteContextHelper.getWorkloadId(), tensorDesc);
        remoteMemoryHelper.clearRemoteMemory();
}

void HDDL2_Remote_Blob_Tests::TearDown() {
    remoteContextPtr = nullptr;
    remoteMemoryHelper.destroyRemoteMemory();
}

//------------------------------------------------------------------------------
TEST_F(HDDL2_Remote_Blob_Tests, CanCreateRemoteBlobUsingContext) {
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);

    ASSERT_NO_THROW(remoteContextPtr->CreateBlob(tensorDesc, blobParams));
}

TEST_F(HDDL2_Remote_Blob_Tests, RemoteBlobFromRemoteMem_WillNotDestroyRemoteMemory_OnDestruction) {
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);

    const std::string memoryData = "Hello there!\n";
    remoteMemoryHelper.getRemoteMemory(memoryData.size());
    remoteMemoryHelper.setRemoteMemory(memoryData);

    { auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams); }

    ASSERT_TRUE(remoteMemoryHelper.isRemoteTheSame(memoryData));
}

TEST_F(HDDL2_Remote_Blob_Tests, CanGetParams) {
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);

    auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams);

    IE::ParamMap params;
    ASSERT_NO_THROW(params = remoteBlobPtr->getParams());
    ASSERT_NE(params.find(IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD)), params.end());
}

TEST_F(HDDL2_Remote_Blob_Tests, CanGetDeviceName) {
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);

    auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams);

    std::string deviceName = remoteBlobPtr->getDeviceName();

    ASSERT_GT(deviceName.size(), 0);
    ASSERT_NE(deviceName.find(pluginName), std::string::npos);
}

TEST_F(HDDL2_Remote_Blob_Tests, CanGetTensorDesc) {
    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);

    auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams);

    IE::TensorDesc resultTensorDesc = remoteBlobPtr->getTensorDesc();

    ASSERT_EQ(resultTensorDesc, tensorDesc);
}

TEST_F(HDDL2_Remote_Blob_Tests, CanChangeRemoteMemory) {
    const std::string memoryData = "Hello from VPUX Plugin!\n";

    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);
    auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams);

    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto memory = lockedMemory.as<char*>();
        memcpy(memory, memoryData.data(), memoryData.size());
    }

    ASSERT_TRUE(remoteMemoryHelper.isRemoteTheSame(memoryData));
}

TEST_F(HDDL2_Remote_Blob_Tests, NonLockedMemoryObject_CanNotChangeRemoteMemory) {
    const std::string memoryData = "Hello from VPUX Plugin!\n";

    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);
    auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams);

    {
        auto memory = remoteBlobPtr->buffer().as<char*>();
        memcpy(memory, memoryData.data(), memoryData.size());
    }

    ASSERT_FALSE(remoteMemoryHelper.isRemoteTheSame(memoryData));
}

TEST_F(HDDL2_Remote_Blob_Tests, MemoryLockedNotInLocalScope_CanNotChangeRemoteMemory) {
    const std::string memoryData = "Hello from VPUX Plugin!\n";

    auto blobParams = RemoteBlob_Helper::wrapRemoteMemFDToMap(remoteMemoryFD);
    auto remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParams);

    auto lockedMemory = remoteBlobPtr->buffer();
    auto memory = lockedMemory.as<char*>();
    memcpy(memory, memoryData.data(), memoryData.size());

    ASSERT_FALSE(remoteMemoryHelper.isRemoteTheSame(memoryData));
}
