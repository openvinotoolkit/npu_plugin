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
#pragma once

#include <tuple>
#include <vector>

namespace vpu {

namespace KmbPlugin {

namespace utils {

class VPUAllocator {
public:
    virtual ~VPUAllocator() = default;
    virtual void* allocate(size_t requestedSize) = 0;
    virtual void* getAllocatedChunkByIndex(size_t chunkIndex) = 0;
    virtual int getFileDescByVirtAddr(void* virtAddr) = 0;
};

class VPUSMMAllocator : public VPUAllocator {
public:
    VPUSMMAllocator(){};
    virtual ~VPUSMMAllocator();
    void* allocate(size_t requestedSize) override;
    void* getAllocatedChunkByIndex(size_t chunkIndex) override;
    int allocateDMA(size_t requestedSize);
    void* importDMA(const int& fileDesc);
    int getFileDescByVirtAddr(void* virtAddr) override;

private:
    std::vector<std::tuple<int, void*, size_t>> _memChunks;
    static uint32_t _pageSize;
};

class NativeAllocator : public VPUAllocator {
public:
    NativeAllocator(){};
    virtual ~NativeAllocator();
    void* allocate(size_t requestedSize) override;
    void* getAllocatedChunkByIndex(size_t chunkIndex) override;
    int getFileDescByVirtAddr(void* virtAddr) override;

private:
    std::vector<uint8_t*> _memChunks;
};

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
