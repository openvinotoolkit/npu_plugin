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

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <hddl2_remote_allocator.h>
#include <ie_blob.h>
#include <unistd.h>

#include <fstream>

using hddl2AllocatorUnitTests = ::testing::Test;

TEST_F(hddl2AllocatorUnitTests, canCreateAllocAndFreeMemory) {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
    ASSERT_NE(devices.size(), 0);
    HddlUnite::Device::Ptr device = std::make_shared<HddlUnite::Device>(devices[0]);
    vpu::HDDL2Plugin::HDDL2RemoteAllocator allocator(device);

    auto size = MAX_ALLOC_SIZE;
    auto handle = allocator.alloc(size);
    ASSERT_NE(handle, nullptr);

    ASSERT_TRUE(allocator.free(handle));
}

TEST_F(hddl2AllocatorUnitTests, canLockAndUnlockMemory) {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
    ASSERT_NE(devices.size(), 0);
    HddlUnite::Device::Ptr device = std::make_shared<HddlUnite::Device>(devices[0]);
    vpu::HDDL2Plugin::HDDL2RemoteAllocator allocator(device);

    auto size = MAX_ALLOC_SIZE;
    auto handle = allocator.alloc(size);
    ASSERT_NE(handle, nullptr);

    auto locked_data = allocator.lock(handle);
    ASSERT_NE(locked_data, nullptr);
    std::memset(locked_data, 0xFF, MAX_ALLOC_SIZE);
    allocator.unlock(handle);

    ASSERT_TRUE(allocator.free(handle));
}

TEST_F(hddl2AllocatorUnitTests, cannotFreeInvalidAddressMemory) {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
    ASSERT_NE(devices.size(), 0);
    HddlUnite::Device::Ptr device = std::make_shared<HddlUnite::Device>(devices[0]);
    vpu::HDDL2Plugin::HDDL2RemoteAllocator allocator(device);

    void* data_ptr = nullptr;

    ASSERT_FALSE(allocator.free(data_ptr));
}

TEST_F(hddl2AllocatorUnitTests, cannotDoDoubleFree) {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
    ASSERT_NE(devices.size(), 0);
    HddlUnite::Device::Ptr device = std::make_shared<HddlUnite::Device>(devices[0]);
    vpu::HDDL2Plugin::HDDL2RemoteAllocator allocator(device);

    auto size = MAX_ALLOC_SIZE;
    auto handle = allocator.alloc(size);
    ASSERT_NE(handle, nullptr);

    ASSERT_TRUE(allocator.free(handle));
    ASSERT_FALSE(allocator.free(handle));
}

TEST_F(hddl2AllocatorUnitTests, cannotRelockData) {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    ASSERT_EQ(code, HddlStatusCode::HDDL_OK);
    ASSERT_NE(devices.size(), 0);
    HddlUnite::Device::Ptr device = std::make_shared<HddlUnite::Device>(devices[0]);
    vpu::HDDL2Plugin::HDDL2RemoteAllocator allocator(device);

    auto size = MAX_ALLOC_SIZE;
    auto handle = allocator.alloc(size);
    ASSERT_NE(handle, nullptr);

    auto locked_data = allocator.lock(handle);
    ASSERT_NE(locked_data, nullptr);

    auto locked_data_oth = allocator.lock(handle);
    ASSERT_EQ(locked_data_oth, nullptr);

    allocator.unlock(handle);

    ASSERT_TRUE(allocator.free(handle));
}
