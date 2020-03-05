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

#include <gtest/gtest.h>
#include <hddl2_helpers/helper_workload_context.h>
#include <hddl2_plugin.h>
#include <hddl2_remote_allocator.h>

#include "hddl2_helpers/helper_remote_memory.h"
#include "helpers/helper_remote_allocator.h"

using namespace vpu::HDDL2Plugin;
using namespace InferenceEngine;

//------------------------------------------------------------------------------
//      class HDDL2_RemoteAllocator_UnitTests Declaration
//------------------------------------------------------------------------------
class HDDL2_RemoteAllocator_UnitTests : public ::testing::Test {
public:
    void SetUp() override;

    HddlUnite::WorkloadContext::Ptr workloadContextPtr = nullptr;

protected:
    WorkloadContext_Helper _workloadContextHelper;
};

//------------------------------------------------------------------------------
//      class HDDL2_RemoteAllocator_UnitTests Implementation
//------------------------------------------------------------------------------
void HDDL2_RemoteAllocator_UnitTests::SetUp() { workloadContextPtr = _workloadContextHelper.getWorkloadContext(); }

//------------------------------------------------------------------------------
//      class HDDL2_RemoteAllocator_UnitTests Initiations - constructors
//------------------------------------------------------------------------------
TEST_F(HDDL2_RemoteAllocator_UnitTests, constructor_CorrectContext_NoThrow) {
    ASSERT_NO_THROW(HDDL2RemoteAllocator allocator(workloadContextPtr));
}

TEST_F(HDDL2_RemoteAllocator_UnitTests, constructor_NullContext_Throw) {
    ASSERT_ANY_THROW(HDDL2RemoteAllocator allocator(nullptr));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteAllocator_UnitTests Initiations - wrapRemoteMemory
//------------------------------------------------------------------------------
// TODO FAIL - HddlUnite problem
TEST_F(HDDL2_RemoteAllocator_UnitTests, DISABLED_wrapRemoteMemory_IncorrectMemoryFD_ReturnNull) {
    auto allocatorPtr = std::make_shared<HDDL2RemoteAllocator>(workloadContextPtr);

    const size_t correctSize = MAX_ALLOC_SIZE;
    const int incorrectMemoryFd = INT32_MAX;

    auto handle = allocatorPtr->wrapRemoteMemory(incorrectMemoryFd, correctSize);
    ASSERT_EQ(handle, nullptr);
}

TEST_F(HDDL2_RemoteAllocator_UnitTests, wrapRemoteMemory_NegativeMemoryFD_ReturnNull) {
    auto allocatorPtr = std::make_shared<HDDL2RemoteAllocator>(workloadContextPtr);

    const size_t correctSize = MAX_ALLOC_SIZE;
    const int negativeMemoryFd = -1;

    auto handle = allocatorPtr->wrapRemoteMemory(negativeMemoryFd, correctSize);
    ASSERT_EQ(handle, nullptr);
}

//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Params
//------------------------------------------------------------------------------
enum RemoteMemoryOwner { IERemoteMemoryOwner = 0, ExternalRemoteMemoryOwner = 1 };
//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Declaration
//------------------------------------------------------------------------------
class HDDL2_Allocator_Manipulations_UnitTests :
    public HDDL2_RemoteAllocator_UnitTests,
    public ::testing::WithParamInterface<RemoteMemoryOwner> {
public:
    void SetUp() override;

    const size_t correctSize = MAX_ALLOC_SIZE;

    Allocator_Helper::Ptr allocatorHelper = nullptr;

    struct PrintToStringParamName {
        std::string operator()(testing::TestParamInfo<RemoteMemoryOwner> const& info) const;
    };
};

//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Implementation
//------------------------------------------------------------------------------
void HDDL2_Allocator_Manipulations_UnitTests::SetUp() {
    HDDL2_RemoteAllocator_UnitTests::SetUp();

    auto owner = GetParam();
    if (owner == IERemoteMemoryOwner) {
        allocatorHelper = std::make_shared<Allocator_CreatedRemoteMemory_Helper>(workloadContextPtr);
    } else {
        allocatorHelper = std::make_shared<Allocator_WrappedRemoteMemory_Helper>(workloadContextPtr);
    }
}

std::string HDDL2_Allocator_Manipulations_UnitTests::PrintToStringParamName::operator()(
    const testing::TestParamInfo<RemoteMemoryOwner>& info) const {
    RemoteMemoryOwner memoryOwner = info.param;
    if (memoryOwner == IERemoteMemoryOwner) {
        return "IERemoteMemoryOwner";
    } else if (memoryOwner == ExternalRemoteMemoryOwner) {
        return "ExternalRemoteMemoryOwner";
    } else {
        return "Unknown params";
    }
}

/**
 * createMemory function hide allocate() call for IE Remote memory owner
 * and wrapRemoteMemory() for External Remote memory owner
 */
//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Initiations - createMemory
//------------------------------------------------------------------------------
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, createMemory_OnCorrectSize_NoThrow) {
    EXPECT_NO_THROW(allocatorHelper->createMemory(correctSize));
}

TEST_P(HDDL2_Allocator_Manipulations_UnitTests, createMemory_OnCorrectSize_NotNullHandle) {
    void* handle = nullptr;
    handle = allocatorHelper->createMemory(correctSize);

    EXPECT_NE(handle, nullptr);
}

TEST_P(HDDL2_Allocator_Manipulations_UnitTests, createMemory_OnIncorrectSize_NullHandle) {
    size_t incorrectSize = MAX_ALLOC_SIZE * 2;

    auto handle = allocatorHelper->createMemory(incorrectSize);
    EXPECT_EQ(handle, nullptr);
}

TEST_P(HDDL2_Allocator_Manipulations_UnitTests, createMemory_OnNegativeSize_NullHandle) {
    size_t negativeSize = -1;

    auto handle = allocatorHelper->createMemory(negativeSize);
    EXPECT_EQ(handle, nullptr);
}

//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Initiations - lock & unlock
//------------------------------------------------------------------------------
// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_canLockAndUnlockMemory) {
    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    auto locked_data = allocator->lock(memoryHandle);

    ASSERT_NE(locked_data, nullptr);
    std::memset(locked_data, 0xFF, MAX_ALLOC_SIZE);

    allocator->unlock(memoryHandle);
}

// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_lock_DoubleLock_SecondReturnNull) {
    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    auto first_locked_data = allocator->lock(memoryHandle);
    ASSERT_NE(first_locked_data, nullptr);

    auto second_locked_data = allocator->lock(memoryHandle);
    ASSERT_EQ(second_locked_data, nullptr);

    allocator->unlock(memoryHandle);
}

// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_unlock_MemoryChanged_RemoteMemoryWillChange) {
    auto owner = GetParam();
    if (owner == IERemoteMemoryOwner) {
        SKIP() << "If Inference Engine own remote memory, we can't get remote memory fd";
    }

    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    auto mem = static_cast<char*>(allocator->lock(memoryHandle));
    const char testValue = 'V';
    mem[0] = testValue;
    allocator->unlock(memoryHandle);

    const std::string remoteMemory = allocatorHelper->getRemoteMemory(sizeof(char));
    const char remoteMemoryValue = remoteMemory[0];

    ASSERT_EQ(remoteMemoryValue, testValue);
}

// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_lock_BeforeUnlock_RemoteMemoryNotChange) {
    auto owner = GetParam();
    if (owner == IERemoteMemoryOwner) {
        SKIP() << "If Inference Engine own remote memory, we can't get remote memory fd";
    }

    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    auto mem = static_cast<char*>(allocator->lock(memoryHandle));
    const char testValue = 'V';
    mem[0] = testValue;

    const std::string remoteMemory = allocatorHelper->getRemoteMemory(sizeof(char));
    const char remoteMemoryValue = remoteMemory[0];
    ASSERT_NE(remoteMemoryValue, testValue);

    allocator->unlock(memoryHandle);
}

//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Initiations - free
//------------------------------------------------------------------------------
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, free_CorrectAddressMemory_ReturnTrue) {
    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    ASSERT_TRUE(allocator->free(memoryHandle));
}

TEST_P(HDDL2_Allocator_Manipulations_UnitTests, free_InvalidAddressMemory_ReturnFalse) {
    auto allocator = allocatorHelper->allocatorPtr;
    void* invalidHandle = nullptr;

    ASSERT_FALSE(allocator->free(invalidHandle));
}

TEST_P(HDDL2_Allocator_Manipulations_UnitTests, free_DoubleCall_ReturnFalseOnSecond) {
    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    ASSERT_TRUE(allocator->free(memoryHandle));
    ASSERT_FALSE(allocator->free(memoryHandle));
}

// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_free_LockedMemory_ReturnFalse) {
    auto memoryHandle = allocatorHelper->createMemory(correctSize);
    auto allocator = allocatorHelper->allocatorPtr;

    allocator->lock(memoryHandle);

    ASSERT_FALSE(allocator->free(memoryHandle));
}

//------------------------------------------------------------------------------
//      class HDDL2_RemoteAllocator_UnitTests Initiations: Functional - memory change
//------------------------------------------------------------------------------
// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_ChangeLocalMemory_RemoteDoesNotChanged) {
    auto allocator = allocatorHelper->allocatorPtr;

    const float testValue = 42.;
    const int size = 100 * sizeof(float);

    auto memoryHandle = allocatorHelper->createMemory(size);
    auto mem = static_cast<float*>(allocator->lock(memoryHandle));

    mem[0] = testValue;

    // Sync memory to device
    allocator->unlock(memoryHandle);

    // Change local memory
    mem[0] = -1;

    // Sync memory from device
    mem = static_cast<float*>(allocator->lock(memoryHandle));

    // Check that same as set
    ASSERT_EQ(testValue, mem[0]);
}

// [Track number: S#28336]
TEST_P(HDDL2_Allocator_Manipulations_UnitTests, DISABLED_ChangeLockedForReadMemory_RemoteDoesNotChanged) {
    auto allocator = allocatorHelper->allocatorPtr;

    const float testValue = 42.;
    const int size = 100 * sizeof(float);

    auto memoryHandle = allocatorHelper->createMemory(size);

    // Set memory on device side
    auto mem = static_cast<float*>(allocator->lock(memoryHandle));
    mem[0] = testValue;
    allocator->unlock(memoryHandle);

    // Get memory for read and change it
    mem = static_cast<float*>(allocator->lock(memoryHandle, InferenceEngine::LOCK_FOR_READ));
    mem[0] = -1;
    allocator->unlock(memoryHandle);

    // Sync memory from device
    mem = static_cast<float*>(allocator->lock(memoryHandle));

    // Check that same as set
    ASSERT_EQ(testValue, mem[0]);
}

//------------------------------------------------------------------------------
//      class HDDL2_Allocator_Manipulations_UnitTests Test case Initiations
//------------------------------------------------------------------------------
const static std::vector<RemoteMemoryOwner> memoryOwners = {IERemoteMemoryOwner, ExternalRemoteMemoryOwner};

INSTANTIATE_TEST_CASE_P(RemoteMemoryOwner, HDDL2_Allocator_Manipulations_UnitTests, ::testing::ValuesIn(memoryOwners),
    HDDL2_Allocator_Manipulations_UnitTests::PrintToStringParamName());
