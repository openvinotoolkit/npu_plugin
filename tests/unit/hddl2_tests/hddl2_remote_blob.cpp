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

#include "hddl2_remote_blob.h"

#include <gtest/gtest.h>

#include "hddl2_helpers/helper_device_name.h"
#include "hddl2_helpers/helper_remote_blob.h"
#include "hddl2_helpers/helper_remote_memory.h"
#include "hddl2_helpers/helper_tensor_description.h"
#include "hddl2_params.hpp"
#include "helper_remote_context.h"
using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Declaration
//------------------------------------------------------------------------------
class HDDL2_RemoteBlob_UnitTests : public testing::Test {
public:
    void SetUp() override;

    const int notExistsBufFd = INT32_MAX;

    InferenceEngine::TensorDesc tensorDesc;
    size_t tensorSize;
    HDDL2RemoteContext::Ptr remoteContextPtr;

    InferenceEngine::ParamMap blobParamMap;
    HDDL2RemoteBlob::Ptr remoteBlobPtr = nullptr;
    const vpu::HDDL2Config config;

    const float value = 42.;
    void setRemoteMemory(const std::string& data);

protected:
    RemoteMemoryFd _remoteMemoryFd = 0;
    TensorDescription_Helper _tensorDescriptionHelper;
    RemoteContext_Helper _remoteContextHelper;
    RemoteMemory_Helper _remoteMemoryHelper;
};

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Implementation
//------------------------------------------------------------------------------
void HDDL2_RemoteBlob_UnitTests::SetUp() {
    tensorDesc = _tensorDescriptionHelper.tensorDesc;
    tensorSize = _tensorDescriptionHelper.tensorSize;

    remoteContextPtr = _remoteContextHelper.remoteContextPtr;
    WorkloadID workloadId = _remoteContextHelper.getWorkloadId();
    _remoteMemoryFd = _remoteMemoryHelper.allocateRemoteMemory(workloadId, tensorSize);

    blobParamMap = RemoteBlob_Helper::wrapRemoteFdToMap(_remoteMemoryFd);
}

void HDDL2_RemoteBlob_UnitTests::setRemoteMemory(const std::string& data) { _remoteMemoryHelper.setRemoteMemory(data); }

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Constructor - tensor + context + params
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_FromCorrectContextAndParams_noException) {
    ASSERT_NO_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, blobParamMap, config));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_FromEmptyParams_ThrowException) {
    InferenceEngine::ParamMap emptyParams = {};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, emptyParams, config));
}

// TODO FAIL - HddlUnite problem
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_constructor_FromNotExistsBufFd_ThrowException) {
    InferenceEngine::ParamMap notExistsFd = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), notExistsBufFd}};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, notExistsFd, config));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_fromIncorrectPararams_ThrowException) {
    InferenceEngine::ParamMap badParams = {{"Bad key", "Bad value"}};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, badParams, config));
}

TEST_F(HDDL2_RemoteBlob_UnitTests, constructor_fromIncorrectType_ThrowException) {
    int incorrectTypeValue = 10;
    InferenceEngine::ParamMap incorrectTypeParams = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD), incorrectTypeValue}};

    ASSERT_ANY_THROW(HDDL2RemoteBlob blob(tensorDesc, remoteContextPtr, incorrectTypeParams, config));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Blob Type checks
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, defaultBlobFromContext_isMemoryBlob_True) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_TRUE(remoteBlobPtr->is<IE::MemoryBlob>());
}

TEST_F(HDDL2_RemoteBlob_UnitTests, defaultBlobFromContext_isRemoteBlob_True) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_TRUE(remoteBlobPtr->is<IE::RemoteBlob>());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations allocate
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, allocate_Default_Works) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_NO_FATAL_FAILURE(remoteBlobPtr->allocate());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations deallocate
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, deallocate_AllocatedBlob_ReturnTrue) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    ASSERT_TRUE(remoteBlobPtr->deallocate());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getDeviceName
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getDeviceName_DeviceAssigned_CorrectName) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    ASSERT_EQ(DeviceName::getNameInPlugin(), remoteBlobPtr->getDeviceName());
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getTensorDesc
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getTensorDesc_AllocatedBlob_ReturnCorrectTensor) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    InferenceEngine::TensorDesc resultTensor = remoteBlobPtr->getTensorDesc();

    ASSERT_EQ(resultTensor, tensorDesc);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations - buffer
//------------------------------------------------------------------------------
// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->buffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_ChangeAllocatedBlob_ShouldStoreNewData) {
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
// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_cbuffer_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->cbuffer();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_buffer_AllocatedBlob_CannotBeChanged) {
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
// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->rwmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rwlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->rwmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

// [Track number: S#28336]
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
// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->rmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_rlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->rmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

// [Track number: S#28336]
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
// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_NotAllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto lockedMemory = remoteBlobPtr->wmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(data, nullptr);
}

// [Track number: S#28336]
TEST_F(HDDL2_RemoteBlob_UnitTests, DISABLED_wlock_AllocatedBlob_ReturnNotNullLocked) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);
    remoteBlobPtr->allocate();

    auto lockedMemory = remoteBlobPtr->wmap();
    auto data = lockedMemory.as<IE::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();

    ASSERT_NE(nullptr, data);
}

// [Track number: S#28336]
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
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto contextPtr = remoteBlobPtr->getContext();

    ASSERT_EQ(remoteContextPtr, contextPtr);
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteBlob_UnitTests Initiations getParams
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteBlob_UnitTests, getParams_AllocatedDefaultBlob_ReturnMapWithParams) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto params = remoteBlobPtr->getParams();

    ASSERT_GE(1, params.size());
}

TEST_F(HDDL2_RemoteBlob_UnitTests, getParams_AllocatedDefaultBlob_SameAsInput) {
    remoteBlobPtr = std::make_shared<HDDL2RemoteBlob>(tensorDesc, remoteContextPtr, blobParamMap, config);

    auto params = remoteBlobPtr->getParams();

    ASSERT_EQ(blobParamMap, params);
}

// TODO We need tests, that on each inference call sync to device not happen. This require
//  mocking allocator.
