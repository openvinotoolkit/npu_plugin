//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>

#include <vpux.hpp>
#include <vpux_backends.h>
#include <vpux_remote_blob.h>
#include <ie_core.hpp>
#include "vpux/utils/IE/blob.hpp"

namespace ie = InferenceEngine;

namespace {
std::vector<uint8_t> fillRemoteBlobWithRandomValues(ie::RemoteBlob::Ptr blob, const size_t offset, const size_t size) {
    if (blob == nullptr) {
        IE_THROW() << "Null blob";
    }
    auto memoryHolder = blob->rwmap();
    uint8_t* blobData = memoryHolder.as<uint8_t*>();
    if (blobData == nullptr) {
        IE_THROW() << "Null data";
    }
    const size_t BYTE_BASE = 256;
    std::generate(blobData + offset, blobData + offset + size, [BYTE_BASE]() {
        return std::rand() % BYTE_BASE;
    });
    return std::vector<uint8_t>(blobData + offset, blobData + offset + size);
}

std::vector<uint8_t> readRemoteBlob(ie::RemoteBlob::Ptr blob, const size_t offset, const size_t size) {
    if (blob == nullptr) {
        IE_THROW() << "Null blob";
    }
    const auto memoryHolder = blob->rmap();
    const uint8_t* blobData = memoryHolder.as<uint8_t*>();
    if (blobData == nullptr) {
        IE_THROW() << "Null data";
    }
    return std::vector<uint8_t>(blobData + offset, blobData + offset + size);
}
}

class VPUXRemoteBlobUnitTests : public ::testing::Test {
protected:
    void SetUp() override;

    vpux::VPUXRemoteContext::Ptr _remoteContext;
    const ie::TensorDesc _tensorDesc = {ie::Precision::FP32, {1, 3, 100, 224}, ie::Layout::NHWC};
    const ie::SizeVector& _dims = _tensorDesc.getDims();
    std::shared_ptr<vpux::Device> _dummyDevice;
    const ie::ParamMap _dummyBlobParams = {{}};
    const size_t _roiOffset = 5;
};

void VPUXRemoteBlobUnitTests::SetUp() {
    vpux::VPUXBackends dummyBackends({"vpu3700_test_backend"});
    _dummyDevice = dummyBackends.getDevice();
    const ie::ParamMap deviceParams = {{}};
    _remoteContext = std::make_shared<vpux::VPUXRemoteContext>(_dummyDevice, deviceParams);
}


// This test is intended for checking cascade ROI case:
// Parent (non-ROI) blob -> ROI blob -> ROI-in-ROI blob
// These blobs use common data from parent blob
// Every ROI blob has its own InferenceEngine::ROI data which keep information about ROI frame geometry (offset from parent and sizes)
// When we are using cascade ROI, ROI offsets are calculated according to the superposition principle
TEST_F(VPUXRemoteBlobUnitTests, cascadeROIBlobCorrect) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const auto fullFrameByteSize = remoteBlobPtr->byteSize();
    const auto fullFrameData = fillRemoteBlobWithRandomValues(remoteBlobPtr, 0, fullFrameByteSize);

    {
        // ROI2 geometry should be {0 + 0, roiOffset + roiOffset, origBlobW, origBlobH - roiOffset - roiOffset}
        // We are using NHWC layout for simply checking ROI-in-ROI blob - it has common part of data
        // with parent (fullFrame) blob from the begin with some offset to the end
        const std::vector<std::size_t> roi1Begin = {0, 0, _roiOffset, 0};
        const std::vector<std::size_t> roi1End = {_dims[0], _dims[1], _dims[2] - _roiOffset, _dims[3]};
        const std::vector<std::size_t> roi2Begin = {0, 0, _roiOffset, 0};
        const std::vector<std::size_t> roi2End = {_dims[0], _dims[1], _dims[2] - 2 * _roiOffset, _dims[3]};
        const ie::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast<ie::RemoteBlob>(remoteBlobPtr->createROI(roi1Begin, roi1End));

        {
            ie::RemoteBlob::Ptr remoteROI2BlobPtr =
                std::static_pointer_cast<ie::RemoteBlob>(remoteROIBlobPtr->createROI(roi2Begin, roi2End));
            vpux::ParsedRemoteBlobParams parsedRemoteBlobParams;
            parsedRemoteBlobParams.update(remoteROI2BlobPtr->getParams());
            const auto roi2Ptr = parsedRemoteBlobParams.getROIPtr();
            ASSERT_NE(roi2Ptr, nullptr);
            ASSERT_EQ(roi2Ptr->posX, roi1Begin[3] + roi2Begin[3]);
            ASSERT_EQ(roi2Ptr->posY, roi1Begin[2] + roi2Begin[2]);
            ASSERT_EQ(roi2Ptr->sizeX, roi2End[3] - roi2Begin[3]);
            ASSERT_EQ(roi2Ptr->sizeY, roi2End[2] - roi2Begin[2]);

            const auto ROIFrameByteSize = remoteROI2BlobPtr->byteSize();
            auto checkROIFrameData = fullFrameData;
            checkROIFrameData.erase(checkROIFrameData.cbegin(), checkROIFrameData.cbegin() + (fullFrameByteSize - ROIFrameByteSize));
            const auto ROIFrameData = readRemoteBlob(remoteROI2BlobPtr, fullFrameByteSize - ROIFrameByteSize, ROIFrameByteSize);
            ASSERT_EQ(checkROIFrameData, ROIFrameData);
        }
    }

    const auto checkFullFrameData = readRemoteBlob(remoteBlobPtr, 0, remoteBlobPtr->byteSize());
    ASSERT_EQ(checkFullFrameData, fullFrameData);
}

TEST_F(VPUXRemoteBlobUnitTests, parentBlobCorrectAfterDeletingROI) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const auto blobDataBefore = readRemoteBlob(remoteBlobPtr, 0, remoteBlobPtr->byteSize());

    {
        const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] - _roiOffset, _dims[2] - _roiOffset};
        auto remoteROIBlobPtr = remoteBlobPtr->createROI(roi);
    }

    const auto blobDataAfter = readRemoteBlob(remoteBlobPtr, 0, remoteBlobPtr->byteSize());
    ASSERT_EQ(blobDataBefore, blobDataAfter);
    ASSERT_NO_THROW(remoteBlobPtr.reset());
}

TEST_F(VPUXRemoteBlobUnitTests, ROIBlobCorrectAfterDeletingParent) {
    ie::RemoteBlob::Ptr remoteROIBlobPtr = nullptr;
    std::vector<uint8_t> blobDataBefore;

    {
        const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] - _roiOffset, _dims[2] - _roiOffset};
        auto remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
        remoteROIBlobPtr = std::static_pointer_cast<ie::RemoteBlob>(remoteBlobPtr->createROI(roi));
        blobDataBefore = readRemoteBlob(remoteROIBlobPtr, 0, remoteROIBlobPtr->byteSize());
    }

    const auto blobDataAfter = readRemoteBlob(remoteROIBlobPtr, 0, remoteROIBlobPtr->byteSize());
    ASSERT_EQ(blobDataBefore, blobDataAfter);
    ASSERT_NO_THROW(remoteROIBlobPtr.reset());
}

TEST_F(VPUXRemoteBlobUnitTests, ROIBlobIntoBoundsNoThrow) {
    auto remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] - _roiOffset, _dims[2] - _roiOffset};
    ASSERT_NO_THROW(remoteBlobPtr->createROI(roi));
}

TEST_F(VPUXRemoteBlobUnitTests, ROIBlobOutOfBoundsThrow) {
    auto remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] + _roiOffset, _dims[2] + _roiOffset};
    ASSERT_ANY_THROW(remoteBlobPtr->createROI(roi));
}

TEST_F(VPUXRemoteBlobUnitTests, multiDimsROIBlobIntoBoundsNoThrow) {
    auto remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const std::vector<std::size_t> roiBegin = {0, 0, _roiOffset, _roiOffset};
    const std::vector<std::size_t> roiEnd = {_dims[0], _dims[1], _dims[2], _dims[3]};
    ASSERT_NO_THROW(remoteBlobPtr->createROI(roiBegin, roiEnd));
}

TEST_F(VPUXRemoteBlobUnitTests, multiDimsROIBlobOutOfBoundsThrow) {
    auto remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const std::vector<std::size_t> roiBegin = {0, 0, _roiOffset, _roiOffset};
    const std::vector<std::size_t> roiEnd = {_dims[0], _dims[1], _dims[3] + _roiOffset, _dims[2] + _roiOffset};
    ASSERT_ANY_THROW(remoteBlobPtr->createROI(roiBegin, roiEnd));
}

TEST_F(VPUXRemoteBlobUnitTests, constructorWithCorrectParamsNoThrow) {
    ASSERT_NO_THROW(vpux::VPUXRemoteBlob blob(_tensorDesc, _remoteContext, _dummyDevice->getAllocator(), _dummyBlobParams));
}

TEST_F(VPUXRemoteBlobUnitTests, constructorWithNullContextThrow) {
    ASSERT_ANY_THROW(vpux::VPUXRemoteBlob blob(_tensorDesc, nullptr, _dummyDevice->getAllocator(), _dummyBlobParams));
}

TEST_F(VPUXRemoteBlobUnitTests, constructorWithNullAllocatorThrow) {
    ASSERT_ANY_THROW(vpux::VPUXRemoteBlob blob(_tensorDesc, _remoteContext, nullptr, _dummyBlobParams));
}

TEST_F(VPUXRemoteBlobUnitTests, getContextAllocatedBlobReturnSameAsOnInit) {
    const auto remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(_tensorDesc, _remoteContext, _dummyDevice->getAllocator(), _dummyBlobParams);
    const auto remoteContextPtr = remoteBlobPtr->getContext();
    ASSERT_EQ(remoteContextPtr, _remoteContext);
}

TEST_F(VPUXRemoteBlobUnitTests, getParamsAllocatedDefaultBlobReturnMapWithParams) {
    const auto remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(_tensorDesc, _remoteContext, _dummyDevice->getAllocator(), _dummyBlobParams);
    const auto params = remoteBlobPtr->getParams();
    ASSERT_GE(params.size(), 1);
}
