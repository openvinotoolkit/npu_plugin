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

#include "vpux_remote_blob.h"

#include <gtest/gtest.h>
#include <hddl2_remote_allocator.h>

#include "hddl2_helpers/helper_device_name.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "vpux/vpux_plugin_params.hpp"
#include "helper_remote_context.h"
#include "memory_usage.h"
#include "skip_conditions.h"
#include "vpux_params.hpp"

using namespace vpux::hddl2;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class HDDL2_RemoteBlob_UnitTests : public testing::Test {
public:
    void SetUp() override;

    const int incorrectWorkloadID = INT32_MAX;

    InferenceEngine::TensorDesc tensorDesc;
    size_t tensorSize;
    vpux::VPUXRemoteContext::Ptr remoteContextPtr;
    std::shared_ptr<vpux::Allocator> allocator;

    InferenceEngine::ParamMap blobParamMap;
    vpux::VPUXRemoteBlob::Ptr remoteBlobPtr = nullptr;

    const float value = 42.;
    void setRemoteMemory(const std::string& data);

protected:
    VpuxRemoteMemoryFD _remoteMemoryFD;
    TensorDescription_Helper _tensorDescriptionHelper;
    RemoteContext_Helper::Ptr _remoteContextHelperPtr = nullptr;
    RemoteMemory_Helper::Ptr _remoteMemoryHelperPtr = nullptr;
};

void HDDL2_RemoteBlob_UnitTests::SetUp() {
    if (HDDL2Backend::isServiceAvailable()) {
        _remoteContextHelperPtr = std::make_shared<RemoteContext_Helper>();
        _remoteMemoryHelperPtr = std::make_shared<RemoteMemory_Helper>();
        auto workloadContextPtr = _remoteContextHelperPtr->getWorkloadContext();
        allocator = std::make_shared<HDDL2RemoteAllocator>(workloadContextPtr);

        tensorDesc = _tensorDescriptionHelper.tensorDesc;
        tensorSize = _tensorDescriptionHelper.tensorSize;

        remoteContextPtr = _remoteContextHelperPtr->remoteContextPtr;
        WorkloadID workloadId = _remoteContextHelperPtr->getWorkloadId();
        _remoteMemoryFD = _remoteMemoryHelperPtr->allocateRemoteMemory(workloadId, tensorDesc);
        blobParamMap = RemoteBlob_Helper::wrapRemoteMemFDToMap(_remoteMemoryFD);
    }
}

void HDDL2_RemoteBlob_UnitTests::setRemoteMemory(const std::string &data) {
    _remoteMemoryHelperPtr->setRemoteMemory(data);
}

//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, CheckRemoteMemoryUsage) {
    SKIP_IF_NO_DEVICE();
#if defined(_WIN32) || defined(__arm__) || defined(__aarch64__)
    SKIP();
#endif

    double vm_before = 0., res_before = 0.;
    MemoryUsage::procMemUsage(vm_before, res_before);
    ASSERT_NE(res_before, 0.);

    const size_t BLOBS_COUNT = 1000000;
    for (size_t cur_blob = 0; cur_blob < BLOBS_COUNT; ++cur_blob) {
        IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    }

    double vm_after = 0., res_after = 0.;
    MemoryUsage::procMemUsage(vm_after, res_after);
    ASSERT_NE(res_after, 0.);

    const double MAX_RES_GROW_KB = 10.;
    ASSERT_LE(res_after - res_before, MAX_RES_GROW_KB);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests - check createROI
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, parentBlobCorrectAfterDeletingROI) {
    SKIP_IF_NO_DEVICE();
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    ASSERT_NE(remoteBlobPtr, nullptr);
    uint8_t *bDataBefore = remoteBlobPtr->rmap().as<uint8_t*>();
    size_t bSizeBefore = remoteBlobPtr->byteSize();
    std::vector<uint8_t> blobDataBefore{bDataBefore, bDataBefore + bSizeBefore};

    {
        IE::ROI roi {0, 2, 2, 221, 221};
        IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));
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

TEST_F(HDDL2_RemoteBlob_UnitTests, ROIBlobCorrectAfterDeletingParent) {
    SKIP_IF_NO_DEVICE();
    IE::RemoteBlob::Ptr remoteROIBlobPtr = nullptr;
    uint8_t *bDataBefore = nullptr;
    size_t bSizeBefore = 0;
    std::vector<uint8_t> blobDataBefore = {};

    {
        IE::ROI roi {0, 2, 2, 221, 221};
        IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
        remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));

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

TEST_F(HDDL2_RemoteBlob_UnitTests, ROIBlobIntoBoundsNoThrow) {
    SKIP_IF_NO_DEVICE();
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    IE::ROI roi {0, 5, 5, 100, 100};
    IE::RemoteBlob::Ptr remoteROIBlobPtr;
    ASSERT_NO_THROW(remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi)));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, ROIBlobOutOfBoundsThrow) {
    SKIP_IF_NO_DEVICE();
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    IE::ROI roi {0, 2, 2, 1000, 1000};
    IE::RemoteBlob::Ptr remoteROIBlobPtr;
    ASSERT_ANY_THROW(remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi)));
}

// This test is intended for checking cascade ROI case:
// Parent (non-ROI) blob -> ROI blob -> ROI-in-ROI blob
// These blobs use common data from parent blob
// Every ROI blob has its own InferenceEngine::ROI data which keep information about ROI frame geometry (offset from parent and sizes)
// When we are using cascade ROI, ROI offsets are calculated according to the superposition principle
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_CascadeROIBlobCorrect) {
    SKIP_IF_NO_DEVICE();
    IE::TensorDesc expectedTensorDesc = {tensorDesc.getPrecision(), tensorDesc.getDims(), IE::Layout::NHWC};
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(expectedTensorDesc, blobParamMap);

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

    const auto origW = tensorDesc.getDims()[3];
    const auto origH = tensorDesc.getDims()[2];
    {
        // ROI2 geometry should be {0 + 0, 1 + 1, origW, origH - 2}
        // We are using NWHC layout for simply checking ROI-in-ROI blob - it has common part of data
        // with parent (fullFrame) blob from the begin with some offset to the end
        IE::ROI roi{0, 0, 1, origW, origH - 1};
        IE::ROI roi2{0, 0, 1, origW, origH - 2};
        IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast<IE::RemoteBlob>(remoteBlobPtr->createROI(roi));

        {
            IE::RemoteBlob::Ptr remoteROI2BlobPtr =
                std::static_pointer_cast<IE::RemoteBlob>(remoteROIBlobPtr->createROI(roi2));
            vpux::ParsedRemoteBlobParams parsedRemoteBlobParams;
            parsedRemoteBlobParams.update(remoteROI2BlobPtr->getParams());
            auto roi2Ptr = parsedRemoteBlobParams.getROIPtr();
            ASSERT_TRUE(roi2Ptr != nullptr);
            ASSERT_TRUE(roi2Ptr->posX == roi.posX + roi2.posX);
            ASSERT_TRUE(roi2Ptr->posY == roi.posY + roi2.posY);
            ASSERT_TRUE(roi2Ptr->sizeX == roi2.sizeX);
            ASSERT_TRUE(roi2Ptr->sizeY == roi2.sizeY);

            size_t ROIFrameByteSize = remoteROI2BlobPtr->byteSize();
            auto checkROIFrameData = fullFrameData;
            checkROIFrameData.erase(checkROIFrameData.cbegin(), checkROIFrameData.cbegin() + (fullFrameByteSize - ROIFrameByteSize));
            uint8_t* ROIFrameRawData = nullptr;
            std::vector<uint8_t> ROIFrameData = {};
            {
                auto memoryHolder = remoteROI2BlobPtr->rmap();
                ROIFrameRawData = memoryHolder.as<uint8_t*>() + (fullFrameByteSize - ROIFrameByteSize);
                ROIFrameData.assign(ROIFrameRawData, ROIFrameRawData + ROIFrameByteSize);
            }
            ASSERT_TRUE(checkROIFrameData == ROIFrameData);
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

    ASSERT_TRUE(checkFullFrameData == fullFrameData);
    ASSERT_TRUE(remoteBlobPtr->deallocate());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Constructor - tensor + context + params
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_FromCorrectContextAndParams_noException) {
    SKIP_IF_NO_DEVICE();
    ASSERT_NO_THROW(vpux::VPUXRemoteBlob blob(tensorDesc, remoteContextPtr, allocator, blobParamMap));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_FromEmptyParams_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap emptyParams = {};

    ASSERT_ANY_THROW(vpux::VPUXRemoteBlob blob(tensorDesc, remoteContextPtr, allocator, emptyParams));
}

// TODO FAIL - HddlUnite problem
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_constructor_FromIncorrectWorkloadID_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap incorrectWorkloadID = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), incorrectWorkloadID}};

    ASSERT_ANY_THROW(vpux::VPUXRemoteBlob blob(tensorDesc, remoteContextPtr, allocator, incorrectWorkloadID));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_fromIncorrectPararams_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap badParams = {{"Bad key", "Bad value"}};

    ASSERT_ANY_THROW(vpux::VPUXRemoteBlob blob(tensorDesc, remoteContextPtr, allocator, badParams));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_fromIncorrectType_ThrowException) {
    SKIP_IF_NO_DEVICE();
    void* incorrectTypeValue = nullptr;
    InferenceEngine::ParamMap incorrectTypeParams = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), incorrectTypeValue}};

    ASSERT_ANY_THROW(vpux::VPUXRemoteBlob blob(tensorDesc, remoteContextPtr, allocator, incorrectTypeParams));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Blob Type checks
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, defaultBlobFromContext_isMemoryBlob_True) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    ASSERT_TRUE(remoteBlobPtr->is<IE::MemoryBlob>());
}

TEST_F(HDDL2_RemoteBlob_UnitTests, defaultBlobFromContext_isRemoteBlob_True) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    ASSERT_TRUE(remoteBlobPtr->is<IE::RemoteBlob>());
}

/** Allocation method doing nothing */
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, allocate_Default_Works) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    ASSERT_NO_FATAL_FAILURE(remoteBlobPtr->allocate());
}

/** Symmetric to allocation, no action should be done */
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, deallocate_AllocatedBlob_ReturnFalse) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    ASSERT_EQ(remoteBlobPtr->deallocate(), false);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, getDeviceName_DeviceAssigned_CorrectName) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    const auto devicesNames = DeviceName::getDevicesNamesWithPrefix();
    auto sameNameFound = devicesNames.find(remoteBlobPtr->getDeviceName());
    EXPECT_TRUE(sameNameFound != devicesNames.end());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getTensorDesc
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getTensorDesc_AllocatedBlob_ReturnCorrectTensor) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    InferenceEngine::TensorDesc resultTensor = remoteBlobPtr->getTensorDesc();

    ASSERT_EQ(resultTensor, tensorDesc);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - buffer
//------------------------------------------------------------------------------

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_NotAllocatedBlob_ReturnNotNullLocked) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto lockedMemory = remoteBlobPtr->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_AllocatedBlob_ReturnNotNullLocked) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_ChangeAllocatedBlob_ShouldStoreNewData) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    // Set some value
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        data[0] = value;
    }
    // Unlock should happen

    // Check that it's the same value
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        ASSERT_EQ(data[0], value);
    }
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - cbuffer
//------------------------------------------------------------------------------

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_cbuffer_NotAllocatedBlob_ReturnNotNullLocked) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto lockedMemory = remoteBlobPtr->cbuffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_AllocatedBlob_CannotBeChanged) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    // Set some value
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        data[0] = value;
    }
    // Unlock should happen

    // Try to change it by calling cbuffer
    {
        auto lockedMemory = remoteBlobPtr->cbuffer();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        data[0] = -1;
    }

    // Check that it's the same as first value
    {
        auto lockedMemory = remoteBlobPtr->buffer();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        ASSERT_EQ(data[0], value);
    }
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - rwlock
//------------------------------------------------------------------------------

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto lockedMemory = remoteBlobPtr->rwmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->rwmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_ChangeAllocatedBlob_ShouldStoreNewData) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    {
        auto lockedMemory = remoteBlobPtr->rwmap();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        data[0] = value;
    }
    {
        auto lockedMemory = remoteBlobPtr->rwmap();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        ASSERT_EQ(data[0], value);
    }
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - rlock
//------------------------------------------------------------------------------

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto lockedMemory = remoteBlobPtr->rmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->rmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_ChangeAllocatedBlob_ShouldNotChangeRemote) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    {
        auto lockedMemory = remoteBlobPtr->rmap();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        data[0] = value;
    }
    {
        auto lockedMemory = remoteBlobPtr->rmap();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        ASSERT_NE(data[0], value);
    }
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - wlock
//------------------------------------------------------------------------------

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto lockedMemory = remoteBlobPtr->wmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->wmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_ChangeAllocatedBlob_ShouldChangeRemote) {
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);
    {
        auto lockedMemory = remoteBlobPtr->wmap();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        data[0] = value;
    }
    {
        auto lockedMemory = remoteBlobPtr->rmap();
        auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        ASSERT_EQ(data[0], value);
    }
}

// TODO FAIL - IE side problem - Not working due to no write lock operation
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_OnLocking_WillNotSyncDataFromDevice) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    const std::string data = "Hello VPUX\n";
    setRemoteMemory(data);

    std::string result;
    {
        auto lockedMemory = remoteBlobPtr->wmap();
        auto lockedData = lockedMemory.as<char*>();
        result = std::string(lockedData);
    }
    ASSERT_NE(result, data);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getContext
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getContext_AllocatedBlob_ReturnSameAsOnInit) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto contextPtr = remoteBlobPtr->getContext();

    ASSERT_EQ(remoteContextPtr, contextPtr);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getParams
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getParams_AllocatedDefaultBlob_ReturnMapWithParams) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto params = remoteBlobPtr->getParams();

    ASSERT_GE(params.size(), 1);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, getParams_AllocatedDefaultBlob_SameAsInput) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<vpux::VPUXRemoteBlob>(tensorDesc, remoteContextPtr, allocator, blobParamMap);

    auto params = remoteBlobPtr->getParams();

    auto memoryFDIter = params.find(IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD));
    ASSERT_NE(memoryFDIter, params.end());
    ASSERT_EQ(memoryFDIter->second.as<VpuxRemoteMemoryFD>(), _remoteMemoryFD);
}

// TODO We need tests, that on each inference call sync to device not happen. This require
//  mocking allocator.
