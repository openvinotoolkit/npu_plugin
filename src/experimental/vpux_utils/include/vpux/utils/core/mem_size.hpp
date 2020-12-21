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

//
// Typed representation for memory sizes.
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

//
// MemType
//

enum class MemType {
    Byte,
    KB,
    MB,
    GB,
};

StringLiteral stringifyEnum(MemType val);

//
// MemMultiplier
//

template <MemType FROM, MemType TO>
struct MemMultiplier;

template <MemType TYPE>
struct MemMultiplier<TYPE, TYPE> final {
    static constexpr size_t value = 1;
};

template <>
struct MemMultiplier<MemType::KB, MemType::Byte> final {
    static constexpr size_t value = 1024;
};
template <>
struct MemMultiplier<MemType::MB, MemType::KB> final {
    static constexpr size_t value = 1024;
};
template <>
struct MemMultiplier<MemType::GB, MemType::MB> final {
    static constexpr size_t value = 1024;
};

template <>
struct MemMultiplier<MemType::MB, MemType::Byte> final {
    static constexpr size_t value =
            MemMultiplier<MemType::MB, MemType::KB>::value * MemMultiplier<MemType::KB, MemType::Byte>::value;
};
template <>
struct MemMultiplier<MemType::GB, MemType::Byte> final {
    static constexpr size_t value =
            MemMultiplier<MemType::GB, MemType::MB>::value * MemMultiplier<MemType::MB, MemType::Byte>::value;
};

template <>
struct MemMultiplier<MemType::GB, MemType::KB> final {
    static constexpr size_t value =
            MemMultiplier<MemType::GB, MemType::MB>::value * MemMultiplier<MemType::MB, MemType::KB>::value;
};

//
// MemSize
//

template <MemType TYPE>
class MemSize final {
public:
    constexpr MemSize() = default;

    constexpr explicit MemSize(uint64_t size): _size(size) {
    }

public:
    template <MemType OTHER>
    constexpr MemSize(const MemSize<OTHER>& size);

    template <MemType OTHER>
    constexpr MemSize<OTHER> to() const;

public:
    constexpr auto count() const {
        return _size;
    }

private:
    uint64_t _size = 0;
};

template <MemType TYPE>
template <MemType OTHER>
constexpr MemSize<TYPE>::MemSize(const MemSize<OTHER>& size): _size(size.count() * MemMultiplier<OTHER, TYPE>::value) {
}

template <MemType TYPE>
template <MemType OTHER>
constexpr MemSize<OTHER> MemSize<TYPE>::to() const {
    return MemSize<OTHER>(*this);
}

using Byte = MemSize<MemType::Byte>;
using MB = MemSize<MemType::MB>;
using KB = MemSize<MemType::KB>;
using GB = MemSize<MemType::GB>;

inline Byte operator""_Byte(unsigned long long const size) {
    return Byte(size);
}
inline MB operator""_MB(unsigned long long const size) {
    return MB(size);
}
inline KB operator""_KB(unsigned long long const size) {
    return KB(size);
}
inline GB operator""_GB(unsigned long long const size) {
    return GB(size);
}

}  // namespace vpux
