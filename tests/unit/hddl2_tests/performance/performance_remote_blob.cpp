//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <gtest/gtest.h>

#include <chrono>

#include "hddl2_helpers/helper_device_name.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "vpux/vpux_plugin_params.hpp"
#include "helper_remote_context.h"
#include "skip_conditions.h"
#include "vpux_remote_blob.h"

using namespace vpux::hddl2;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_PerformanceTests Declaration
//------------------------------------------------------------------------------
class HDDL2_RemoteBlob_PerformanceTests : public testing::Test {
public:
    void SetUp() override;

    InferenceEngine::TensorDesc tensorDesc;
    size_t tensorSize;
    vpux::VPUXRemoteContext::Ptr remoteContextPtr;

    InferenceEngine::ParamMap blobParamMap;
    vpux::VPUXRemoteBlob::Ptr remoteBlobPtr = nullptr;

    void setRemoteMemory(const std::string& data);

protected:
    VpuxRemoteMemoryFD _remoteMemoryFD = -1;
    TensorDescription_Helper _tensorDescriptionHelper;
    RemoteContext_Helper::Ptr _remoteContextHelperPtr;
    RemoteMemory_Helper::Ptr _remoteMemoryHelperPtr;
};

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_PerformanceTests Implementation
//------------------------------------------------------------------------------
void HDDL2_RemoteBlob_PerformanceTests::SetUp() {
    if (HDDL2Backend::isServiceAvailable()) {
        _remoteContextHelperPtr = std::make_shared<RemoteContext_Helper>();
        _remoteMemoryHelperPtr = std::make_shared<RemoteMemory_Helper>();
        tensorDesc = _tensorDescriptionHelper.tensorDesc;
        tensorSize = _tensorDescriptionHelper.tensorSize;

        remoteContextPtr = _remoteContextHelperPtr->remoteContextPtr;
        WorkloadID workloadId = _remoteContextHelperPtr->getWorkloadId();
        _remoteMemoryFD = _remoteMemoryHelperPtr->allocateRemoteMemory(workloadId, tensorDesc);

        blobParamMap = RemoteBlob_Helper::wrapRemoteMemFDToMap(_remoteMemoryFD);
    }
}

void HDDL2_RemoteBlob_PerformanceTests::setRemoteMemory(const std::string &data) {
    _remoteMemoryHelperPtr->setRemoteMemory(data);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_PerformanceTests - check performance
//------------------------------------------------------------------------------
// TODO create separate target for performance tests
// TODO Investigate performance degradation on blob creation Track: #-40443
TEST_F(HDDL2_RemoteBlob_PerformanceTests, DISABLED_createRemoteBlobPerformance) {
    SKIP_IF_NO_DEVICE();
    const size_t BLOBS_COUNT = 1000000;
    auto start_time = std::chrono::steady_clock::now();
    for (size_t cur_blob = 0; cur_blob < BLOBS_COUNT; ++cur_blob) {
        IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
#ifdef __unix__
    const double MAX_SPENT_TIME = 1.0;
#else
    const double MAX_SPENT_TIME = 2.0;
#endif
    ASSERT_LE(elapsed_seconds.count(), MAX_SPENT_TIME);
}

// TODO Investigate performance degradation on blob creation Track: #-40443
TEST_F(HDDL2_RemoteBlob_PerformanceTests, DISABLED_createROIBlobPerformance) {
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
    const double MAX_SPENT_TIME = 2.0;
    ASSERT_LE(elapsed_seconds.count(), MAX_SPENT_TIME);
}
