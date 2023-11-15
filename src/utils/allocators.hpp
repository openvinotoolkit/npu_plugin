//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
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
    VPUSMMAllocator(const int deviceId = 0);
    virtual ~VPUSMMAllocator();
    void* allocate(size_t requestedSize) override;
    void* getAllocatedChunkByIndex(size_t chunkIndex) override;
    int allocateDMA(size_t requestedSize);
    void* importDMA(const int& fileDesc);
    int getFileDescByVirtAddr(void* virtAddr) override;

    bool free(void* handle);

private:
    std::vector<std::tuple<int, void*, size_t>> _memChunks;
    static uint32_t _pageSize;
    int _deviceId;
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
