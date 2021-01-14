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

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <climits>

namespace vpux {

//
// MemType
//

enum class MemType {
    Bit,
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

template <>
struct MemMultiplier<MemType::Byte, MemType::Bit> final {
    static constexpr int64_t value = CHAR_BIT;
};
template <>
struct MemMultiplier<MemType::KB, MemType::Byte> final {
    static constexpr int64_t value = 1024;
};
template <>
struct MemMultiplier<MemType::MB, MemType::KB> final {
    static constexpr int64_t value = 1024;
};
template <>
struct MemMultiplier<MemType::GB, MemType::MB> final {
    static constexpr int64_t value = 1024;
};

template <>
struct MemMultiplier<MemType::MB, MemType::Byte> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::MB, MemType::KB>::value * MemMultiplier<MemType::KB, MemType::Byte>::value;
};
template <>
struct MemMultiplier<MemType::MB, MemType::Bit> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::MB, MemType::Byte>::value * MemMultiplier<MemType::Byte, MemType::Bit>::value;
};

template <>
struct MemMultiplier<MemType::GB, MemType::KB> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::GB, MemType::MB>::value * MemMultiplier<MemType::MB, MemType::KB>::value;
};
template <>
struct MemMultiplier<MemType::GB, MemType::Byte> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::GB, MemType::KB>::value * MemMultiplier<MemType::KB, MemType::Byte>::value;
};
template <>
struct MemMultiplier<MemType::GB, MemType::Bit> final {
    static constexpr int64_t value =
            MemMultiplier<MemType::GB, MemType::Byte>::value * MemMultiplier<MemType::Byte, MemType::Bit>::value;
};

//
// MemSize
//

namespace details {

template <MemType FROM, MemType TO, typename = void>
struct HasMemMultiplier {
    static constexpr bool value = false;
};

template <MemType FROM, MemType TO>
struct HasMemMultiplier<FROM, TO, std::enable_if_t<MemMultiplier<FROM, TO>::value != 0>> {
    static constexpr bool value = true;
};

}  // namespace details

template <MemType TYPE>
class MemSize final {
public:
    constexpr MemSize() = default;
    constexpr MemSize(const MemSize&) = default;

    constexpr explicit MemSize(int64_t size): _size(size) {
    }

public:
    template <MemType OTHER>
    constexpr MemSize(const MemSize<OTHER>& size, require_t<details::HasMemMultiplier<OTHER, TYPE>>* = nullptr)
            : _size(size.count() * MemMultiplier<OTHER, TYPE>::value) {
    }

    template <MemType OTHER>
    MemSize(const MemSize<OTHER>& size, require_t<details::HasMemMultiplier<TYPE, OTHER>>* = nullptr) {
        constexpr auto mult = MemMultiplier<TYPE, OTHER>::value;

        VPUX_THROW_UNLESS(size.count() % mult == 0, "Can't convert {0} {1} to {2}", size.count(), OTHER, TYPE);
        _size = size.count() / mult;
    }

    template <class OTHER>
    constexpr OTHER to() const {
        return OTHER(*this);
    }

public:
    constexpr auto count() const {
        return _size;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "{0} {1}", count(), TYPE);
    }

private:
    int64_t _size = 0;
};

template <MemType TYPE>
MemSize<TYPE> operator+(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return MemSize<TYPE>(size1.count() + size2.count());
}

template <MemType TYPE>
MemSize<TYPE> operator*(MemSize<TYPE> size, int64_t mult) {
    return MemSize<TYPE>(size.count() * mult);
}
template <MemType TYPE>
MemSize<TYPE> operator*(int64_t mult, MemSize<TYPE> size) {
    return MemSize<TYPE>(size.count() * mult);
}

template <MemType TYPE>
bool operator==(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() == size2.count();
}
template <MemType TYPE>
bool operator!=(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() != size2.count();
}
template <MemType TYPE>
bool operator>(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() > size2.count();
}
template <MemType TYPE>
bool operator>=(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() >= size2.count();
}
template <MemType TYPE>
bool operator<(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() < size2.count();
}
template <MemType TYPE>
bool operator<=(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() <= size2.count();
}

template <MemType TYPE>
int64_t operator%(MemSize<TYPE> size1, MemSize<TYPE> size2) {
    return size1.count() % size2.count();
}

using Bit = MemSize<MemType::Bit>;
using Byte = MemSize<MemType::Byte>;
using KB = MemSize<MemType::KB>;
using MB = MemSize<MemType::MB>;
using GB = MemSize<MemType::GB>;

inline constexpr Bit operator""_Bit(unsigned long long const size) {
    return Bit(static_cast<int64_t>(size));
}
inline constexpr Byte operator""_Byte(unsigned long long const size) {
    return Byte(static_cast<int64_t>(size));
}
inline constexpr KB operator""_KB(unsigned long long const size) {
    return KB(static_cast<int64_t>(size));
}
inline constexpr MB operator""_MB(unsigned long long const size) {
    return MB(static_cast<int64_t>(size));
}
inline constexpr GB operator""_GB(unsigned long long const size) {
    return GB(static_cast<int64_t>(size));
}

}  // namespace vpux
