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

#include <list>
#include <vector>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/containers.hpp>
#include <vpu/model/data.hpp>

namespace vpu {


//
// Common allocation constants
//

const int DDR_MAX_SIZE = 512 * 1024 * 1024;
const int CMX_SLICE_SIZE = 128 * 1024;
const int DATA_ALIGNMENT = 64;

//
// Allocator Structs
//

namespace allocator {

struct MemChunk final {
    MemoryType memType = MemoryType::DDR;
    int pointer = 0;
    int offset = 0;
    int size = 0;
    int inUse = 0;

    std::list<MemChunk>::iterator _posInList;
};

struct FreeMemory final {
    int offset = 0;
    int size = 0;
};

struct MemoryPool final {
    int curMemOffset = 0;
    int memUsed = 0;
    std::list<MemChunk> allocatedChunks;
    SmallVector<FreeMemory> freePool;

    void clear() {
        curMemOffset = 0;
        memUsed = 0;
        allocatedChunks.clear();
        freePool.clear();
    }
};

}  // namespace allocator


}  // namespace vpu
