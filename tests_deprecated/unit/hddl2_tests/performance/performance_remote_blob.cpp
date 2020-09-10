//
// Copyright 2020 Intel Corporation.
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

#include "hddl2_remote_blob.h"

#include <chrono>
#include <gtest/gtest.h>

#include "hddl2_helpers/helper_device_name.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "helper_remote_context.h"
#include "skip_conditions.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_PerformanceTests Declaration
//------------------------------------------------------------------------------
class HDDL2_RemoteBlob_PerformanceTests : public testing::Test {
public:
    void SetUp() override;

    InferenceEngine::TensorDesc tensorDesc;
    size_t tensorSize;
    HDDL2RemoteContext::Ptr remoteContextPtr;

    InferenceEngine::ParamMap blobParamMap;
    HDDL2RemoteBlob::Ptr remoteBlobPtr = nullptr;
    const vpu::HDDL2Config config = vpu::HDDL2Config();

    void setRemoteMemory(const std::string& data);

protected:
    RemoteMemoryFd _remoteMemoryFd = 0;
    TensorDescription_Helper _tensorDescriptionHelper;
    RemoteContext_Helper::Ptr _remoteContextHelperPtr;
    RemoteMemory_Helper::Ptr _remoteMemoryHelperPtr;
};

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_PerformanceTests Implementation
//------------------------------------------------------------------------------
void HDDL2_RemoteBlob_PerformanceTests::SetUp() {
    if (HDDL2Metrics::isServiceAvailable()) {
        _remoteContextHelperPtr = std::make_shared<RemoteContext_Helper>();
        _remoteMemoryHelperPtr = std::make_shared<RemoteMemory_Helper>();
        tensorDesc = _tensorDescriptionHelper.tensorDesc;
        tensorSize = _tensorDescriptionHelper.tensorSize;

        remoteContextPtr = _remoteContextHelperPtr->remoteContextPtr;
        WorkloadID workloadId = _remoteContextHelperPtr->getWorkloadId();
        _remoteMemoryFd = _remoteMemoryHelperPtr->allocateRemoteMemory(workloadId, tensorSize);

        blobParamMap = RemoteBlob_Helper::wrapRemoteFdToMap(_remoteMemoryFd);
    }
}

void HDDL2_RemoteBlob_PerformanceTests::setRemoteMemory(const std::string &data) {
    _remoteMemoryHelperPtr->setRemoteMemory(data);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_PerformanceTests - check performance
//------------------------------------------------------------------------------
// TODO create separate target for performance tests
TEST_F(HDDL2_RemoteBlob_PerformanceTests, createRemoteBlobPerformance) {
    SKIP_IF_NO_DEVICE();
    const size_t BLOBS_COUNT = 1000000;
    auto start_time = std::chrono::steady_clock::now();
    for (size_t cur_blob = 0; cur_blob < BLOBS_COUNT; ++cur_blob) {
        IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    const double MAX_SPENT_TIME = 1.0;
    ASSERT_LE(elapsed_seconds.count(), MAX_SPENT_TIME);
}

TEST_F(HDDL2_RemoteBlob_PerformanceTests, createROIBlobPerformance) {
    SKIP_IF_NO_DEVICE();
    IE::ROI roi {0, 2, 2, 221, 221};
    const size_t BLOBS_COUNT = 1000000;
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    auto start_time = std::chrono::steady_clock::now();
    for (size_t cur_blob = 0; cur_blob < BLOBS_COUNT; ++cur_blob) {
        IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    const double MAX_SPENT_TIME = 0.5;
    ASSERT_LE(elapsed_seconds.count(), MAX_SPENT_TIME);
}
