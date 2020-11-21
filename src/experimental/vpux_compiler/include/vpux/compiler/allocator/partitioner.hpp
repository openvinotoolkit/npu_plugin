//
// Copyright 2019-2020 Intel Corporation.
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

//
// Partitioner finds and allocates unused portion of memory from the contiguous
// memory array; returns the portion back after their usage is finished
//

#pragma once

#include <limits>
#include <vector>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace vpux {

using AddressType = uint64_t;
constexpr AddressType InvalidAddress = std::numeric_limits<AddressType>::max();

class Partitioner final {
public:
    enum class Direction { Up, Down };

    struct Gap final {
        AddressType begin;
        AddressType end;

        AddressType size() const {
            assert(end > begin);
            return end - begin;
        }
    };

public:
    explicit Partitioner(AddressType totalSize);

public:
    AddressType alloc(AddressType size, AddressType alignment = 1, Direction dir = Direction::Up);

    // Called when client is certain about ability to allocate at the specified
    // addr
    void allocFixed(AddressType addr, AddressType size);

    void free(AddressType addr, AddressType size);

public:
    AddressType totalSize() const {
        return _totalSize;
    }

    AddressType totalFreeSize() const;

    AddressType maxFreeSize() const;

    const std::vector<Gap>& gaps() const {
        return _gaps;
    }

public:
    static bool intersects(AddressType addr1, AddressType size1, AddressType addr2, AddressType size2);

private:
    AddressType getAddrFromGap(size_t pos, AddressType size, AddressType alignment, Direction dir);
    AddressType useGap(size_t pos, AddressType alignedBegin, AddressType size);
    AddressType chooseMinimalGap(AddressType size, AddressType alignment, Direction dir);

private:
    std::vector<Gap> _gaps;
    AddressType _totalSize = 0;
};

}  // namespace vpux
