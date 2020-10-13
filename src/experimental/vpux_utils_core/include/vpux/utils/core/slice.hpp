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

#include <cassert>
#include <cstdint>

namespace vpux {

class Slice final {
public:
    Slice() = default;

    Slice(int64_t begin, int64_t end) : _begin(begin), _end(end) {
        assert(end >= begin);
    }

public:
    int64_t begin() const {
        return _begin;
    }

    int64_t end() const {
        return _end;
    }

    int64_t length() const {
        return end() - begin();
    }

public:
    bool intersects(const Slice& other) const {
        return (begin() < other.end()) && (other.begin() < end());
    }

    bool contains(const Slice& other) const {
        return (begin() <= other.begin()) && (_end >= other.end());
    }

    // Represents `this` range as sub-slice of the `parent`.
    Slice asSubSlice(const Slice& parent) const {
        assert(parent.contains(*this));

        return {begin() - parent.begin(), end() - parent.begin()};
    }

private:
    int64_t _begin = 0;
    int64_t _end = 0;
};

inline bool operator==(const Slice& s1, const Slice& s2) {
    return s1.begin() == s2.begin() && s1.end() == s2.end();
}
inline bool operator!=(const Slice& s1, const Slice& s2) {
    return !(s1 == s2);
}

}  // namespace vpux
