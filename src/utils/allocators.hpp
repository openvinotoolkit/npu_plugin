//
// Copyright 2019 Intel Corporation.
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
