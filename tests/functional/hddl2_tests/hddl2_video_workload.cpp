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

#include "RemoteMemory.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_helpers/models/precompiled_resnet.h"
#include "hddl2_params.hpp"
#include "ie_core.hpp"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_VideoWorkload_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_VideoWorkload_Tests : public ::testing::Test {};

//------------------------------------------------------------------------------
//      class VideoPipeline Declaration
//------------------------------------------------------------------------------
class VideoPipeline {
public:
    /**
     * 1. Create workload context
     * 2. Create remote memory and set some information to it
     * 3. Save workloadContextId and dmaFd to reuse it as parameters for IE
     */
    bool createVideoPipeline(WorkloadID& workloadContextID, uint64_t& remoteMemoryFd, const std::string& data);

    ~VideoPipeline();

protected:
    // TODO Use real input instead
    const size_t _inputSize = 200000;

    HddlUnite::SMM::RemoteMemory::Ptr _remoteMemoryPtr = nullptr;
    WorkloadID _workloadId = -1;
};

//------------------------------------------------------------------------------
//      class VideoPipeline Implementation
//------------------------------------------------------------------------------
bool VideoPipeline::createVideoPipeline(
    WorkloadID& workloadContextID, uint64_t& remoteMemoryFd, const std::string& data) {
    auto context = HddlUnite::createWorkloadContext();

    auto ret = context->setContext(_workloadId);
    if (ret != HDDL_OK) {
        printf("Error: WorkloadContext set context failed");
        return false;
    }
    ret = registerWorkloadContext(context);
    if (ret != HDDL_OK) {
        printf("Error: WorkloadContext register on WorkloadCache failed");
        return false;
    }

    _remoteMemoryPtr = HddlUnite::SMM::allocate(*context, _inputSize);

    if (_remoteMemoryPtr == nullptr) {
        return false;
    }

    _remoteMemoryPtr->syncToDevice(data.data(), data.size());

    workloadContextID = _workloadId;
    remoteMemoryFd = _remoteMemoryPtr->getDmaBufFd();

    return true;
}

VideoPipeline::~VideoPipeline() { HddlUnite::unregisterWorkloadContext(_workloadId); }

//------------------------------------------------------------------------------
//      class HDDL2_VideoWorkload_Tests Initiation
//------------------------------------------------------------------------------
/**
 * 1. Create remote blob in video pipeline and set data to it
 * 2. Create remote context from workload id
 * 3. Create remote blob from dma fd
 * 4. Check that input for inference the same as video pipeline provided
 */
TEST_F(HDDL2_VideoWorkload_Tests, CanGetInputFromCreatedVideoPipeline) {
    const std::string data_str = "Hello HDDL2 Plugin";
    WorkloadID workloadContextID;
    uint64_t remoteMemoryFd;

    // ---- Create video pipeline mock
    VideoPipeline videoPipeline;
    ASSERT_TRUE(videoPipeline.createVideoPipeline(workloadContextID, remoteMemoryFd, data_str));

    // ---- Load inference engine instance
    InferenceEngine::Core ie;

    // ---- Init context map and create context based on it
    IE::ParamMap paramMap = {{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadContextID}};
    IE::RemoteContext::Ptr contextPtr = ie.CreateContext("HDDL2", paramMap);

    // ---- Create remote blob by using already exists fd
    IE::ParamMap blobParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd}};

    // TODO This information should me granted from executable network
    TensorDescription_Helper tensorDescriptionHelper;
    auto tensorDesc = tensorDescriptionHelper.tensorDesc;
    IE::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(tensorDesc, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    // ---- Check that memory contains message we are expecting
    remoteBlobPtr->allocate();

    std::string first_output;
    // TODO Any other way to unlock memory? (destructor of locked memory should be called)
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto data = lockedMemory.as<char*>();
        first_output = std::string(data);
    }
    ASSERT_EQ(data_str, first_output);
}
