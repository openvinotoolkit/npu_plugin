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
#include "hddl2_helpers/helper_device_emulator.h"
#include "hddl2_helpers/helper_workload_context.h"

using namespace HddlUnite;

//------------------------------------------------------------------------------
//      class HDDL2_Hddl_Unite_Tests
//------------------------------------------------------------------------------
class HDDL2_Hddl_Unite_Tests : public ::testing::Test {
public:
    WorkloadContext_Helper workloadContextHelper;
};

//------------------------------------------------------------------------------
//      class HDDL2_Hddl_Unite_Tests Initiation - construct
//------------------------------------------------------------------------------

// TODO FAIL - HddlUnite problem
TEST_F(HDDL2_Hddl_Unite_Tests, DISABLED_WrapIncorrectFd_ThrowException) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    const size_t incorrectFd = INT32_MAX;

    ASSERT_ANY_THROW(SMM::RemoteMemory wrappedRemoteMemory(*workloadContext, incorrectFd, 1));
}

TEST_F(HDDL2_Hddl_Unite_Tests, WrapSameSize_NoException) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    const size_t size = 100;

    SMM::RemoteMemory::Ptr remoteMemoryPtr = SMM::allocate(*workloadContext, size);

    ASSERT_NO_THROW(SMM::RemoteMemory wrappedRemoteMemory(*workloadContext, remoteMemoryPtr->getDmaBufFd(), size));
}

TEST_F(HDDL2_Hddl_Unite_Tests, WrapSmallerSize_NoException) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    const size_t size = 100;
    const size_t smallerSizeToWrap = 10;

    SMM::RemoteMemory::Ptr remoteMemoryPtr = SMM::allocate(*workloadContext, size);

    ASSERT_NO_THROW(
        SMM::RemoteMemory wrappedRemoteMemory(*workloadContext, remoteMemoryPtr->getDmaBufFd(), smallerSizeToWrap));
}

// TODO FAIL - HddlUnite problem
TEST_F(HDDL2_Hddl_Unite_Tests, DISABLED_WrapBiggerSize_ThrowException) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    const size_t size = 100;
    const size_t biggerSizeToWrap = size * 10;

    SMM::RemoteMemory::Ptr remoteMemoryPtr = SMM::allocate(*workloadContext, size);

    ASSERT_ANY_THROW(
        SMM::RemoteMemory wrappedRemoteMemory(*workloadContext, remoteMemoryPtr->getDmaBufFd(), biggerSizeToWrap));
}

// TODO FAIL - HdllUnite problem
TEST_F(HDDL2_Hddl_Unite_Tests, DISABLED_WrapNegativeFd_ThrowException) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();

    ASSERT_ANY_THROW(SMM::RemoteMemory wrappedRemoteMemory(*workloadContext, -1, 1));
}

//------------------------------------------------------------------------------
//      class HDDL2_Hddl_Unite_Tests Initiation - getters
//------------------------------------------------------------------------------
TEST_F(HDDL2_Hddl_Unite_Tests, CanGetAvailableDevices) {
    std::vector<HddlUnite::Device> devices;

    HddlStatusCode code = getAvailableDevices(devices);
    std::cout << " [INFO] - Devices found: " << devices.size() << std::endl;

    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
}

//------------------------------------------------------------------------------
//      class HDDL2_Hddl_Unite_Tests Initiation - Change
//------------------------------------------------------------------------------
TEST_F(HDDL2_Hddl_Unite_Tests, CanCreateAndChangeRemoteMemory) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    const std::string message = "Hello there\n";

    const size_t size = 100;

    SMM::RemoteMemory::Ptr remoteMemoryPtr = SMM::allocate(*workloadContext, size);

    { remoteMemoryPtr->syncToDevice(message.data(), message.size()); }

    char resultData[size] = {};
    { remoteMemoryPtr->syncFromDevice(resultData, size); }

    const std::string resultMessage(resultData);

    ASSERT_EQ(resultData, message);
}

TEST_F(HDDL2_Hddl_Unite_Tests, WrappedMemoryWillHaveSameData) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    const std::string message = "Hello there\n";

    const size_t size = 100;

    SMM::RemoteMemory::Ptr remoteMemoryPtr = SMM::allocate(*workloadContext, size);
    { remoteMemoryPtr->syncToDevice(message.data(), message.size()); }

    // Wrapped memory
    SMM::RemoteMemory wrappedRemoteMemory(*workloadContext, remoteMemoryPtr->getDmaBufFd(), size);
    char resultData[size] = {};
    { wrappedRemoteMemory.syncFromDevice(resultData, size); }

    const std::string resultMessage(resultData);

    ASSERT_EQ(resultData, message);
}

TEST_F(HDDL2_Hddl_Unite_Tests, CanSetAndGetRemoteContextUsingId) {
    auto workloadContext = workloadContextHelper.getWorkloadContext();
    WorkloadID workloadId = workloadContext->getWorkloadContextID();

    // Get workload context and check that name the same
    {
        auto workload_context = HddlUnite::queryWorkloadContext(workloadId);
        EXPECT_NE(nullptr, workload_context);

        const std::string deviceName = workload_context->getDevice()->getName();
        EXPECT_EQ(deviceName, emulatorDeviceName);
    }

    // Destroy after finishing working with context
    HddlUnite::unregisterWorkloadContext(workloadId);
}

TEST_F(HDDL2_Hddl_Unite_Tests, QueryIncorrectWorkloadIdReturnNull) {
    auto workload_context = HddlUnite::queryWorkloadContext(INT32_MAX);
    ASSERT_EQ(workload_context, nullptr);
}

TEST_F(HDDL2_Hddl_Unite_Tests, CanCreateTwoDifferentContextOneAfterAnother) {
    // Destory default remote context from SetUp
    workloadContextHelper.destroyHddlUniteContext(workloadContextHelper.getWorkloadId());

    WorkloadID firstWorkloadId = WorkloadContext_Helper::createAndRegisterWorkloadContext();
    unregisterWorkloadContext(firstWorkloadId);

    WorkloadID secondWorkloadId = WorkloadContext_Helper::createAndRegisterWorkloadContext();
    unregisterWorkloadContext(secondWorkloadId);
    ASSERT_NE(firstWorkloadId, secondWorkloadId);
}

// TODO FAIL - HUNG - HddlUnite problem
TEST_F(HDDL2_Hddl_Unite_Tests, DISABLED_CreatingTwoWorkloadContextForSameProcessWillReturnError) {
    // First context
    WorkloadID firstWorkloadId = -1;
    auto firstContext = HddlUnite::createWorkloadContext();

    auto ret = firstContext->setContext(firstWorkloadId);
    EXPECT_EQ(ret, HDDL_OK);
    ret = registerWorkloadContext(firstContext);
    EXPECT_EQ(ret, HDDL_OK);

    // First context
    WorkloadID secondWorkloadId = -1;
    auto secondContext = HddlUnite::createWorkloadContext();

    ret = secondContext->setContext(secondWorkloadId);
    EXPECT_EQ(ret, HDDL_GENERAL_ERROR);
    ret = registerWorkloadContext(secondContext);
    EXPECT_EQ(ret, HDDL_GENERAL_ERROR);

    EXPECT_NE(firstWorkloadId, secondWorkloadId);

    HddlUnite::unregisterWorkloadContext(firstWorkloadId);
    HddlUnite::unregisterWorkloadContext(secondWorkloadId);
}
