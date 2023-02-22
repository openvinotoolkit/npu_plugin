//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/range.hpp"

#include <cstdint>

namespace vpux {

//
// Mask
//

// Packed mask for resources availability marks.

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

//
// std::hash specialization
//

namespace std {

template <>
struct hash<vpux::Mask> final {
    size_t operator()(vpux::Mask mask) const {
        return static_cast<size_t>(mask.code());
    }
};

}  // namespace std

//
// llvm::format_provider specialization
//

namespace llvm {

template <>
struct format_provider<vpux::Mask> final {
    static void format(const vpux::Mask& mask, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(mask.asRange()).format(stream, style);
    }
};

}  // namespace llvm
