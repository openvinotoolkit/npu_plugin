//
// Copyright 2021 Intel Corporation.
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

#include <vpux.hpp>
#include <vpux_backends.h>
#include <vpux_remote_blob.h>
#include <ie_core.hpp>
#include "vpux/utils/IE/blob.hpp"

namespace ie = InferenceEngine;

class VPUXRemoteBlobUnitTests : public ::testing::Test {
    void SetUp() override {
        vpux::VPUXBackends dummyBackends({"vpu3700_test_backend"});
        _dummyDevice = dummyBackends.getDevice();
        const ie::ParamMap deviceParams = {{}};
        _remoteContext = std::make_shared<vpux::VPUXRemoteContext>(_dummyDevice, deviceParams);
    }

public:
    vpux::VPUXRemoteContext::Ptr _remoteContext;
    const ie::TensorDesc _tensorDesc = {ie::Precision::FP32, {1, 3, 100, 224}, ie::Layout::NHWC};
    const ie::SizeVector& _dims = _tensorDesc.getDims();
    std::shared_ptr<vpux::Device> _dummyDevice;
    const ie::ParamMap _dummyBlobParams = {{}};
    const size_t _roiOffset = 5;
};

// This test is intended for checking cascade ROI case:
// Parent (non-ROI) blob -> ROI blob -> ROI-in-ROI blob
// These blobs use common data from parent blob
// Every ROI blob has its own InferenceEngine::ROI data which keep information about ROI frame geometry (offset from parent and sizes)
// When we are using cascade ROI, ROI offsets are calculated according to the superposition principle
TEST_F(VPUXRemoteBlobUnitTests, cascadeROIBlobCorrect) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);

    size_t fullFrameByteSize = remoteBlobPtr->byteSize();
    uint8_t* fullFrameRawData = nullptr;
    std::vector<uint8_t> fullFrameData = {};
    {
        auto memoryHolder = remoteBlobPtr->rwmap();
        fullFrameRawData = memoryHolder.as<uint8_t*>();
        const size_t BYTE_BASE = 256;
        std::generate(fullFrameRawData, fullFrameRawData + fullFrameByteSize, [BYTE_BASE]() {
            return std::rand() % BYTE_BASE;
        });
        fullFrameData.assign(fullFrameRawData, fullFrameRawData + fullFrameByteSize);
    }

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
            ASSERT_TRUE(roi2Ptr != nullptr);
            ASSERT_TRUE(roi2Ptr->posX == roi1Begin[3] + roi2Begin[3]);
            ASSERT_TRUE(roi2Ptr->posY == roi1Begin[2] + roi2Begin[2]);
            ASSERT_TRUE(roi2Ptr->sizeX == roi2End[3] - roi2Begin[3]);
            ASSERT_TRUE(roi2Ptr->sizeY == roi2End[2] - roi2Begin[2]);

            const size_t ROIFrameByteSize = remoteROI2BlobPtr->byteSize();
            auto checkROIFrameData = fullFrameData;
            checkROIFrameData.erase(checkROIFrameData.cbegin(), checkROIFrameData.cbegin() + (fullFrameByteSize - ROIFrameByteSize));
            uint8_t* ROIFrameRawData = nullptr;
            std::vector<uint8_t> ROIFrameData = {};
            {
                auto memoryHolder = remoteROI2BlobPtr->rmap();
                ROIFrameRawData = memoryHolder.as<uint8_t*>() + (fullFrameByteSize - ROIFrameByteSize);
                ROIFrameData.assign(ROIFrameRawData, ROIFrameRawData + ROIFrameByteSize);
            }
            ASSERT_EQ(checkROIFrameData, ROIFrameData);
        }
    }

    size_t checkFullFrameByteSize = remoteBlobPtr->byteSize();
    uint8_t* checkFullFrameRawData = nullptr;
    std::vector<uint8_t> checkFullFrameData= {};
    {
        auto memoryHolder = remoteBlobPtr->rmap();
        checkFullFrameRawData = memoryHolder.as<uint8_t*>();
        checkFullFrameData.assign(checkFullFrameRawData, checkFullFrameRawData + checkFullFrameByteSize);
    }

    ASSERT_EQ(checkFullFrameData, fullFrameData);
}

TEST_F(VPUXRemoteBlobUnitTests, parentBlobCorrectAfterDeletingROI) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    ASSERT_NE(remoteBlobPtr, nullptr);
    const uint8_t *bDataBefore = remoteBlobPtr->rmap().as<uint8_t*>();
    const size_t bSizeBefore = remoteBlobPtr->byteSize();
    const std::vector<uint8_t> blobDataBefore{bDataBefore, bDataBefore + bSizeBefore};

    {
        const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] - _roiOffset, _dims[2] - _roiOffset};
        auto remoteROIBlobPtr = remoteBlobPtr->createROI(roi);
    }

    uint8_t *bDataAfter = nullptr;
    size_t bSizeAfter = 0;
    std::vector<uint8_t> blobDataAfter = {};
    ASSERT_NO_THROW(bDataAfter = remoteBlobPtr->rmap().as<uint8_t*>());
    ASSERT_NO_THROW(bSizeAfter = remoteBlobPtr->byteSize());
    ASSERT_NO_THROW(blobDataAfter.assign(bDataAfter, bDataAfter + bSizeAfter));
    ASSERT_TRUE(blobDataBefore == blobDataAfter);
    ASSERT_NO_THROW(remoteBlobPtr.reset());
}

TEST_F(VPUXRemoteBlobUnitTests, ROIBlobCorrectAfterDeletingParent) {
    ie::RemoteBlob::Ptr remoteROIBlobPtr = nullptr;
    uint8_t *bDataBefore = nullptr;
    size_t bSizeBefore = 0;
    std::vector<uint8_t> blobDataBefore = {};

    {
        const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] - _roiOffset, _dims[2] - _roiOffset};
        auto remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
        remoteROIBlobPtr = std::static_pointer_cast<ie::RemoteBlob>(remoteBlobPtr->createROI(roi));
        bDataBefore = remoteROIBlobPtr->rmap().as<uint8_t*>();
        bSizeBefore = remoteROIBlobPtr->byteSize();
        blobDataBefore.assign(bDataBefore, bDataBefore + bSizeBefore);
    }

    uint8_t *bDataAfter = nullptr;
    size_t bSizeAfter = 0;
    std::vector<uint8_t> blobDataAfter = {};
    ASSERT_NO_THROW(bDataAfter = remoteROIBlobPtr->rmap().as<uint8_t*>());
    ASSERT_NO_THROW(bSizeAfter = remoteROIBlobPtr->byteSize());
    ASSERT_NO_THROW(blobDataAfter.assign(bDataAfter, bDataAfter + bSizeAfter));
    ASSERT_TRUE(blobDataBefore == blobDataAfter);
    ASSERT_NO_THROW(remoteROIBlobPtr.reset());
}

TEST_F(VPUXRemoteBlobUnitTests, ROIBlobIntoBoundsNoThrow) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] - _roiOffset, _dims[2] - _roiOffset};
    ASSERT_NO_THROW(remoteBlobPtr->createROI(roi));
}

TEST_F(VPUXRemoteBlobUnitTests, ROIBlobOutOfBoundsThrow) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const ie::ROI roi {0, _roiOffset, _roiOffset, _dims[3] + _roiOffset, _dims[2] + _roiOffset};
    ASSERT_ANY_THROW(remoteBlobPtr->createROI(roi));
}

TEST_F(VPUXRemoteBlobUnitTests, multiDimsROIBlobIntoBoundsNoThrow) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
    const std::vector<std::size_t> roiBegin = {0, 0, _roiOffset, _roiOffset};
    const std::vector<std::size_t> roiEnd = {_dims[0], _dims[1], _dims[2], _dims[3]};
    ASSERT_NO_THROW(remoteBlobPtr->createROI(roiBegin, roiEnd));
}

TEST_F(VPUXRemoteBlobUnitTests, multiDimsROIBlobOutOfBoundsThrow) {
    ie::RemoteBlob::Ptr remoteBlobPtr = _remoteContext->CreateBlob(_tensorDesc, _dummyBlobParams);
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
