//
// Copyright 2020 Intel Corporation.
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
