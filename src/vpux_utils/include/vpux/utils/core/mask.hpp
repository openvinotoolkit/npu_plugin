//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <cstdint>

namespace vpux {

//
// Mask
//

// Packed mask for resouces availability marks.

class Mask {
public:
    using StorageType = uint32_t;
    static constexpr size_t NUM_BITS = sizeof(StorageType) * 8;

public:
    static Mask fromCode(StorageType code);
    static Mask fromCount(int32_t count);
    static Mask fromRange(int32_t start, int32_t end);
    static Mask fromIndexes(ArrayRef<int32_t> indexes);

public:
    size_t size() const;

    int32_t operator[](size_t ind) const;

public:
    auto asRange() const {
        return irange(size()) | transformed([this](size_t ind) {
                   return (*this)[ind];
               });
    }

public:
    bool isContinous() const;

public:
    StorageType code() const {
        return _code;
    }

private:
    StorageType _code = 0;
};

inline bool operator==(Mask m1, Mask m2) {
    return m1.code() == m2.code();
}
inline bool operator!=(Mask m1, Mask m2) {
    return m1.code() != m2.code();
}

}  // namespace vpux
