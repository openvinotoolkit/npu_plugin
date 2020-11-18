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
// Helper functions for hash calculation.
//

#pragma once

#include "vpux/utils/core/mask.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <functional>
#include <tuple>

namespace vpux {

namespace details {

size_t combineHashVals(size_t seed, size_t val);

}  // namespace details

template <typename T>
size_t getHash(const T& val) {
    return std::hash<T>()(val);
}

template <typename T, typename... Args>
size_t getHash(const T& val, Args&&... args) {
    return details::combineHashVals(getHash(val),
                                    getHash(std::forward<Args>(args)...));
}

template <class Range>
size_t getRangeHash(const Range& r) {
    size_t seed = 0;
    for (const auto& val : r) {
        seed = details::combineHashVals(seed, getHash(val));
    }
    return seed;
}

}  // namespace vpux

namespace std {

template <>
struct hash<vpux::StringRef> final {
    size_t operator()(vpux::StringRef str) const;
};

template <>
struct hash<vpux::Mask> final {
    size_t operator()(vpux::Mask mask) const {
        return static_cast<size_t>(mask.code());
    }
};

template <typename T1, typename T2>
struct hash<pair<T1, T2>> final {
    size_t operator()(const pair<T1, T2>& p) const {
        return vpux::getHash(p.first, p.second);
    }
};

template <typename T>
struct hash<vpux::Optional<T>> final {
    size_t operator()(const vpux::Optional<T>& opt) const {
        return opt.has_value() ? vpux::getHash(opt.value()) : 0;
    }
};

template <typename... Args>
struct hash<tuple<Args...>> final {
    size_t operator()(const tuple<Args...>& val) const {
        size_t seed = 0;
        hashItems(val, seed);
        return seed;
    }

private:
    template <size_t Index = 0>
    static auto hashItems(const tuple<Args...>&, size_t&)
            -> enable_if_t<Index == sizeof...(Args)> {
    }

    template <size_t Index = 0>
            static auto hashItems(const tuple<Args...>& val, size_t& seed)
                    -> enable_if_t < Index<sizeof...(Args)> {
        seed = vpux::details::combineHashVals(seed,
                                              vpux::getHash(get<Index>(val)));
        hashItems<Index + 1>(val, seed);
    }
};

}  // namespace std
