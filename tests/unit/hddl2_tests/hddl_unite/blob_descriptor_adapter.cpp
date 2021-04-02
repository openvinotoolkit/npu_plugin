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

#include "blob_descriptor_adapter.h"

#include <gtest/gtest.h>
#include <hddl2_helpers/helper_remote_blob.h>
#include <hddl2_helpers/helper_remote_memory.h>
#include <hddl2_helpers/helper_workload_context.h>
#include <hddl2_remote_allocator.h>
#include <helper_remote_context.h>
#include <skip_conditions.h>

#include <ie_core.hpp>

#include "hddl2_params.hpp"

using namespace vpux::hddl2;
namespace IE = InferenceEngine;

class BlobDescriptorAdapter_UnitTests: public ::testing::Test {
public:
    const unsigned width = 224;
    const unsigned height = 224;
    const bool isInput = true;

    IE::RemoteBlob::CPtr createRemoteBlob(const IE::TensorDesc& tensorDesc);

protected:
    HddlUnite::RemoteMemory::Ptr _remoteMemory = nullptr;
    RemoteContext_Helper::Ptr _remoteContextHelperPtr = nullptr;
    RemoteMemory_Helper::Ptr _remoteMemoryHelperPtr = nullptr;
};

IE::RemoteBlob::CPtr BlobDescriptorAdapter_UnitTests::createRemoteBlob(const IE::TensorDesc& tensorDesc) {
    _remoteContextHelperPtr = std::make_shared<RemoteContext_Helper>();
    _remoteMemoryHelperPtr = std::make_shared<RemoteMemory_Helper>();
    auto remoteContextPtr = _remoteContextHelperPtr->remoteContextPtr;

    WorkloadID workloadId = _remoteContextHelperPtr->getWorkloadId();
    // Size this blob will be not used, size doesn't matter
    const size_t size = 100;
    _remoteMemory = _remoteMemoryHelperPtr->allocateRemoteMemory(workloadId, size);

    const auto blobParamMap = RemoteBlob_Helper::wrapRemoteMemToMap(_remoteMemory);
    return remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
}

TEST_F(BlobDescriptorAdapter_UnitTests, Correct_HeightWidth) {
    BlobDescType blobType = BlobDescType::ImageWorkload; // Only ImageWorkload for local blobs
    IE::TensorDesc tensorDesc(IE::Precision::U8, {1,1, height,width}, IE::Layout::NCHW);
    IE::DataPtr blobDesc = std::make_shared<IE::Data>("inputBlob", tensorDesc);
    const auto blob = IE::make_shared_blob<uint8_t>(tensorDesc);
    blob->allocate();

    BlobDescriptorAdapter adapter(blobType, blobDesc, IE::BGR, isInput);
    adapter.createUniteBlobDesc(isInput);
    // Information about frame size will be not specified until update call with blob
    const auto hddlUniteBlobDesc = adapter.updateUniteBlobDesc(blob);

    EXPECT_EQ(hddlUniteBlobDesc.m_resWidth, width);
    EXPECT_EQ(hddlUniteBlobDesc.m_resHeight, height);
}

// // TODO [Track number: S#43588]
TEST_F(BlobDescriptorAdapter_UnitTests, DISABLED_Correct_HeightWidth_LocalROI) {
    BlobDescType blobType = BlobDescType::ImageWorkload; // Only ImageWorkload for local blobs
    const IE::ROI roi(0, 2, 2, 200, 200);

    IE::TensorDesc tensorDesc(IE::Precision::U8, {1,1,height,width}, IE::Layout::NCHW);
    IE::DataPtr blobDesc = std::make_shared<IE::Data>("inputBlob", tensorDesc);
    const auto blob = IE::make_shared_blob<uint8_t>(tensorDesc);
    blob->allocate();
    const auto roiBlob = blob->createROI(roi);

    BlobDescriptorAdapter adapter(blobType, blobDesc, IE::BGR, isInput);
    adapter.createUniteBlobDesc(isInput);
    // Information about frame size will be not specified until update call with blob
    const auto hddlUniteBlobDesc = adapter.updateUniteBlobDesc(roiBlob);

    EXPECT_EQ(hddlUniteBlobDesc.m_resWidth, width);
    EXPECT_EQ(hddlUniteBlobDesc.m_resHeight, height);
}

TEST_F(BlobDescriptorAdapter_UnitTests, Correct_HeightWidth_RemoteROI) {
    SKIP_IF_NO_DEVICE();
    BlobDescType blobType = BlobDescType::VideoWorkload;
    const IE::ROI roi(0, 2, 2, 200, 200);

    IE::TensorDesc tensorDesc(IE::Precision::U8, {1,1,height,width}, IE::Layout::NCHW);
    IE::DataPtr blobDesc = std::make_shared<IE::Data>("inputBlob", tensorDesc);
    auto blob = createRemoteBlob(tensorDesc);
    const auto roiBlob = blob->createROI(roi);

    BlobDescriptorAdapter adapter(blobType, blobDesc, IE::BGR, isInput);
    adapter.createUniteBlobDesc(isInput);
    // Information about frame size will be not specified until update call with blob
    const auto hddlUniteBlobDesc = adapter.updateUniteBlobDesc(roiBlob);

    EXPECT_EQ(hddlUniteBlobDesc.m_resWidth, width);
    EXPECT_EQ(hddlUniteBlobDesc.m_resHeight, height);
}

