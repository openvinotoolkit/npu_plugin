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

#include <unordered_set>
#include <list>
#include <vector>

#include <vpu/utils/enums.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/allocator/structs.hpp>

namespace vpu {

//
// AllocatorForShaves
//

class AllocatorForShaves final {
public:
    explicit AllocatorForShaves(allocator::MemoryPool &cmxMemoryPool);

    void reset();

    bool allocateSHAVEs(
                const Stage& stage,
                StageSHAVEsRequirements reqs);
    void freeSHAVEs();

    int getLockedSHAVEs() { return _lockedSHAVEs; }

    void selfCheck();

private:
    int _lockedSHAVEs = 0;

    allocator::MemoryPool &_cmxMemoryPool;
};

}  // namespace vpu
