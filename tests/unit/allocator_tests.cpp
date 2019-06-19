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

#include <ie_blob.h>

#include "kmb_allocator.h"

using namespace testing;
using namespace vpu::KmbPlugin;

class kmbAllocatorUnitTests : public ::testing::Test {

};

TEST(kmbAllocatorUnitTests, canAllocatePageSizeAlignedMemorySegment) {
    KmbAllocator allocator;

    long pageSize = getpagesize();

    size_t alignedSize = 2 * pageSize;
    ASSERT_NE(allocator.alloc(alignedSize), nullptr);
}

TEST(kmbAllocatorUnitTests, canAllocateNotPageSizeAlignedMemorySegment) {
    KmbAllocator allocator;

    long pageSize = getpagesize();

    size_t notAlignedSize = 2 * pageSize - 1;
    ASSERT_NE(allocator.alloc(notAlignedSize), nullptr);
}

TEST(kmbAllocatorUnitTests, canFreeMemory) {
    KmbAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    ASSERT_TRUE(allocator.free(data));
}

TEST(kmbAllocatorUnitTests, canWriteAndReadAllocatedMemory) {
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

TEST(kmbAllocatorUnitTests, cannotFreeInvalidAddressMemory) {
    KmbAllocator allocator;

    auto data = nullptr;

    ASSERT_FALSE(allocator.free(data));
}

TEST(kmbAllocatorUnitTests, cannotDoDoubleFree) {
    KmbAllocator allocator;

    size_t size = 10;
    auto data = allocator.alloc(size);
    ASSERT_NE(data, nullptr);

    ASSERT_TRUE(allocator.free(data));
    ASSERT_FALSE(allocator.free(data));
}

TEST(kmbAllocatorUnitTests, canCreateBlobBasedOnAllocator) {
    const std::shared_ptr<InferenceEngine::IAllocator> customAllocator(new KmbAllocator());

    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, customAllocator);

    ASSERT_NE(blob, nullptr);
}

TEST(kmbAllocatorUnitTests, canWriteToBlobMemory) {
    const std::shared_ptr<InferenceEngine::IAllocator> customAllocator(new KmbAllocator());

    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {1, 1, 1, 1}, InferenceEngine::Layout::NCHW);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, customAllocator);

    ASSERT_NE(blob, nullptr);

    blob->allocate();
    blob->buffer().as<uint8_t *>()[0] = 0;
}