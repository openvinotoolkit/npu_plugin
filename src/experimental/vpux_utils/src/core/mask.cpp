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

#include "vpux/utils/core/mask.hpp"

#include <cassert>

using namespace vpux;

//
// Mask
//

constexpr size_t vpux::Mask::NUM_BITS;

Mask vpux::Mask::fromCode(StorageType code) {
    Mask mask;
    mask._code = code;
    return mask;
}

Mask vpux::Mask::fromCount(int32_t count) {
    assert(count >= 0);
    assert(static_cast<size_t>(count) <= NUM_BITS);

    Mask mask;
    mask._code = static_cast<StorageType>((1 << count) - 1);
    return mask;
}

Mask vpux::Mask::fromRange(int32_t start, int32_t end) {
    assert(start >= 0 && end >= 0 && start <= end);
    assert(static_cast<size_t>(end) <= NUM_BITS);

    Mask mask;
    mask._code = static_cast<StorageType>(((1 << (end - start)) - 1) << start);
    return mask;
}

Mask vpux::Mask::fromIndexes(ArrayRef<int32_t> indexes) {
    Mask mask;
    for (auto ind : indexes) {
        assert(ind >= 0);
        assert(static_cast<size_t>(ind) < NUM_BITS);

        mask._code |= static_cast<StorageType>(1 << (ind + 1));
    }
    return mask;
}

size_t vpux::Mask::size() const {
    size_t res = 0;

    for (auto temp = _code; temp > 0; temp >>= 1) {
        if (temp & 1) {
            ++res;
        }
    }

    return res;
}

int32_t vpux::Mask::operator[](size_t ind) const {
    assert(ind < size());

    int32_t res = 0;

    for (auto temp = _code; temp > 0; temp >>= 1, ++res) {
        if (temp & 1) {
            if (ind == 0) {
                return res;
            }

            --ind;
        }
    }

    return -1;
}

bool vpux::Mask::isContinous() const {
    bool metSet = false;
    bool metUnsetAfterSet = false;

    for (auto temp = _code; temp > 0; temp >>= 1) {
        if ((temp & 1) && metUnsetAfterSet) {
            return false;
        }

        if ((temp & 1) == 0 && metSet) {
            metUnsetAfterSet = true;
        }

        if (temp & 1) {
            metSet = true;
        }
    }

    return true;
}
