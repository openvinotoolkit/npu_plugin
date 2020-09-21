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

#include <gtest/gtest.h>

#include "hddl2_helpers/helper_device_name.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "helper_remote_context.h"
#include "memory_usage.h"
#include "skip_conditions.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class HDDL2_RemoteBlob_UnitTests : public testing::Test {
public:
    void SetUp() override;

    const int incorrectWorkloadID = INT32_MAX;

    InferenceEngine::TensorDesc tensorDesc;
    size_t tensorSize;
    HDDL2RemoteContext::Ptr remoteContextPtr;

    InferenceEngine::ParamMap blobParamMap;
    HDDL2RemoteBlob::Ptr remoteBlobPtr = nullptr;
    const vpu::HDDL2Config config = vpu::HDDL2Config();

    const float value = 42.;
    void setRemoteMemory(const std::string& data);

protected:
    HddlUnite::RemoteMemory::Ptr _remoteMemory;
    TensorDescription_Helper _tensorDescriptionHelper;
    RemoteContext_Helper::Ptr _remoteContextHelperPtr = nullptr;
    RemoteMemory_Helper::Ptr _remoteMemoryHelperPtr = nullptr;
};

void HDDL2_RemoteBlob_UnitTests::SetUp() {
    if (HDDL2Metrics::isServiceAvailable()) {
        _remoteContextHelperPtr = std::make_shared<RemoteContext_Helper>();
        _remoteMemoryHelperPtr = std::make_shared<RemoteMemory_Helper>();

        tensorDesc = _tensorDescriptionHelper.tensorDesc;
        tensorSize = _tensorDescriptionHelper.tensorSize;

        remoteContextPtr = _remoteContextHelperPtr->remoteContextPtr;
        WorkloadID workloadId = _remoteContextHelperPtr->getWorkloadId();
        _remoteMemory = _remoteMemoryHelperPtr->allocateRemoteMemory(workloadId, tensorSize);

        blobParamMap = RemoteBlob_Helper::wrapRemoteMemToMap(_remoteMemory);
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
    ASSERT_TRUE(remoteBlobPtr->deallocate());
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
    ASSERT_TRUE(remoteROIBlobPtr->deallocate());
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

TEST_F(HDDL2_RemoteBlob_UnitTests, CascadeROIBlobCorrect) {
    SKIP_IF_NO_DEVICE();
    IE::RemoteBlob::Ptr remoteBlobPtr = remoteContextPtr->CreateBlob(tensorDesc, blobParamMap);
    uint8_t *bDataBefore = remoteBlobPtr->rmap().as<uint8_t*>();
    size_t bSizeBefore = remoteBlobPtr->byteSize();
    std::vector<uint8_t> blobDataBefore{bDataBefore, bDataBefore + bSizeBefore};

    {
        IE::ROI roi {0, 2, 2, 221, 221};
        IE::ROI roi2 {0, 5, 5, 100, 100};
        IE::RemoteBlob::Ptr remoteROIBlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteBlobPtr->createROI(roi));

        {
            IE::RemoteBlob::Ptr remoteROI2BlobPtr = std::static_pointer_cast <IE::RemoteBlob> (remoteROIBlobPtr->createROI(roi2));
            auto roi2Ptr = remoteROI2BlobPtr->as<HDDL2RemoteBlob>()->getROIPtr();
            ASSERT_TRUE(roi2Ptr != nullptr);
            ASSERT_TRUE(roi2Ptr->posX == roi.posX + roi2.posX);
            ASSERT_TRUE(roi2Ptr->posY == roi.posY + roi2.posY);
            ASSERT_TRUE(roi2Ptr->sizeX == roi2.sizeX);
            ASSERT_TRUE(roi2Ptr->sizeY == roi2.sizeY);
            uint8_t *bROIData = remoteROI2BlobPtr->rmap().as<uint8_t*>();
            size_t bROISize = remoteROI2BlobPtr->byteSize();
            std::vector<uint8_t> blobROIData{bROIData, bROIData + bROISize};
            ASSERT_TRUE(blobDataBefore == blobROIData);
        }
    }

    uint8_t *bDataAfter = nullptr;
    size_t bSizeAfter = 0;
    std::vector<uint8_t> blobDataAfter = {};
    ASSERT_NO_THROW(bDataAfter = remoteBlobPtr->rmap().as<uint8_t*>());
    ASSERT_NO_THROW(bSizeAfter = remoteBlobPtr->byteSize());
    ASSERT_NO_THROW(blobDataAfter.assign(bDataAfter, bDataAfter + bSizeAfter));
    ASSERT_TRUE(blobDataBefore == blobDataAfter);
    ASSERT_TRUE(remoteBlobPtr->deallocate());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Constructor - tensor + context + params
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_FromCorrectContextAndParams_noException) {
    SKIP_IF_NO_DEVICE();
    ASSERT_NO_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, blobParamMap, config));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_FromEmptyParams_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap emptyParams = {};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, emptyParams, config));
}

// TODO FAIL - HddlUnite problem
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_constructor_FromIncorrectWorkloadID_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap incorrectWorkloadID = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), incorrectWorkloadID}};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, incorrectWorkloadID, config));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_fromIncorrectPararams_ThrowException) {
    SKIP_IF_NO_DEVICE();
    InferenceEngine::ParamMap badParams = {{"Bad key", "Bad value"}};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, badParams, config));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_fromIncorrectType_ThrowException) {
    SKIP_IF_NO_DEVICE();
    int incorrectTypeValue = 10;
    InferenceEngine::ParamMap incorrectTypeParams = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), incorrectTypeValue}};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, incorrectTypeParams, config));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Blob Type checks
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, defaultBlobFromContext_isMemoryBlob_True) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_TRUE(remoteBlobPtr->is<IE::MemoryBlob>());
}

TEST_F(HDDL2_RemoteBlob_UnitTests, defaultBlobFromContext_isRemoteBlob_True) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_TRUE(remoteBlobPtr->is<IE::RemoteBlob>());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations allocate
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, allocate_Default_Works) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_NO_FATAL_FAILURE(remoteBlobPtr->allocate());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations deallocate
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, deallocate_AllocatedBlob_ReturnTrue) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    ASSERT_TRUE(remoteBlobPtr->deallocate());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getDeviceName
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getDeviceName_DeviceAssigned_CorrectName) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_EQ(DeviceName::getNameInPlugin(), remoteBlobPtr->getDeviceName());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getTensorDesc
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getTensorDesc_AllocatedBlob_ReturnCorrectTensor) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    InferenceEngine::TensorDesc resultTensor = remoteBlobPtr->getTensorDesc();

    ASSERT_EQ(resultTensor, tensorDesc);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - buffer
//------------------------------------------------------------------------------

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_NotAllocatedBlob_ReturnNotNullLocked) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_AllocatedBlob_ReturnNotNullLocked) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_ChangeAllocatedBlob_ShouldStoreNewData) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->cbuffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_AllocatedBlob_CannotBeChanged) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->rwmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->rwmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_ChangeAllocatedBlob_ShouldStoreNewData) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->rmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->rmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_ChangeAllocatedBlob_ShouldNotChangeRemote) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->wmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->wmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_ChangeAllocatedBlob_ShouldChangeRemote) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    const std::string data = "Hello HDDL2\n";
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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto contextPtr = remoteBlobPtr->getContext();

    ASSERT_EQ(remoteContextPtr, contextPtr);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getParams
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getParams_AllocatedDefaultBlob_ReturnMapWithParams) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto params = remoteBlobPtr->getParams();

    ASSERT_GE(1, params.size());
}

TEST_F(HDDL2_RemoteBlob_UnitTests, getParams_AllocatedDefaultBlob_SameAsInput) {
    SKIP_IF_NO_DEVICE();
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto params = remoteBlobPtr->getParams();

    ASSERT_EQ(blobParamMap, params);
}

// TODO We need tests, that on each inference call sync to device not happen. This require
//  mocking allocator.
