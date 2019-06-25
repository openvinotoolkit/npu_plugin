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
#include <unistd.h>
#include <cstring>
#include <fstream>

#include <ie_blob.h>

#include "kmb_allocator.h"


using namespace testing;
using namespace vpu::KmbPlugin;

class kmbAllocatorUnitTests : public ::testing::Test {
protected:
    void SetUp() override {
        std::ifstream modulesLoaded("/proc/modules");
        std::string line;
        while (std::getline(modulesLoaded, line))
        {
            if (line.find("vpusmm_driver") != std::string::npos) {
                isVPUSMMDriverFound = true;
                break;
            }
        }
    }

    bool isVPUSMMDriverFound = false;
};

TEST_F(kmbAllocatorUnitTests, canAllocatePageSizeAlignedMemorySegment) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbAllocator allocator;

    long pageSize = getpagesize();

    size_t alignedSize = 2 * pageSize;
    ASSERT_NE(allocator.alloc(alignedSize), nullptr);
}

TEST_F(kmbAllocatorUnitTests, canAllocateNotPageSizeAlignedMemorySegment) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbAllocator allocator;

    long pageSize = getpagesize();

    size_t notAlignedSize = 2 * pageSize - 1;
    ASSERT_NE(allocator.alloc(notAlignedSize), nullptr);
}

TEST_F(kmbAllocatorUnitTests, canFreeMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    ASSERT_TRUE(allocator.free(data));
}

TEST_F(kmbAllocatorUnitTests, canWriteAndReadAllocatedMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbAllocator allocator;

    size_t size = 10;
    void *data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    // this memory should be accessible for manipulations
    char *lockedData = reinterpret_cast<char *>(allocator.lock(data, InferenceEngine::LockOp::LOCK_FOR_WRITE));

    const int MAGIC_NUMBER = 0x13;
    std::memset(lockedData, MAGIC_NUMBER, size);

    std::vector<char> actual(size);
    std::memcpy(actual.data(), data, size);
    ASSERT_TRUE(std::count(actual.begin(), actual.end(), MAGIC_NUMBER) == size);
}

TEST_F(kmbAllocatorUnitTests, cannotFreeInvalidAddressMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbAllocator allocator;

    auto data = nullptr;

    ASSERT_FALSE(allocator.free(data));
}

TEST_F(kmbAllocatorUnitTests, cannotDoDoubleFree) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    KmbAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    ASSERT_TRUE(allocator.free(data));
    ASSERT_FALSE(allocator.free(data));
}

TEST_F(kmbAllocatorUnitTests, canCreateBlobBasedOnAllocator) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }

    const std::shared_ptr<InferenceEngine::IAllocator> customAllocator(new KmbAllocator());

    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, customAllocator);

    ASSERT_NE(blob, nullptr);
}

TEST_F(kmbAllocatorUnitTests, canWriteToBlobMemory) {
    if (!isVPUSMMDriverFound) {
        SKIP() << "vpusmm_driver not found. Please install before running tests";
    }
    
    const std::shared_ptr<InferenceEngine::IAllocator> customAllocator(new KmbAllocator());

    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, customAllocator);

    ASSERT_NE(blob, nullptr);

    blob->allocate();
    blob->buffer().as<uint8_t *>()[0] = 0;
}