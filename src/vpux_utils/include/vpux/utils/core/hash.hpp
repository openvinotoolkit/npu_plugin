//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Helper functions for hash calculation.
//

#pragma once

#include "vpux/utils/core/type_traits.hpp"

#include <llvm/ADT/Hashing.h>

#include <functional>
#include <tuple>

namespace vpux {

template <typename T>
size_t getHash(const T& val) {
    return std::hash<T>()(val);
}

template <typename T, typename... Args>
size_t getHash(const T& val, Args&&... args) {
    return llvm::hash_combine(getHash(val), getHash(std::forward<Args>(args)...));
}

template <class Range>
size_t getRangeHash(const Range& r) {
    return llvm::hash_combine_range(r.begin(), r.end());
}

}  // namespace vpux

//
// std::hash specialization
//

namespace std {

template <typename T1, typename T2>
struct hash<pair<T1, T2>> final {
    size_t operator()(const pair<T1, T2>& p) const {
        return vpux::getHash(p.first, p.second);
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
    static auto hashItems(const tuple<Args...>&, size_t&) -> enable_if_t<Index == sizeof...(Args)> {
    }

    template <size_t Index = 0>
            static auto hashItems(const tuple<Args...>& val, size_t& seed) -> enable_if_t < Index<sizeof...(Args)> {
        seed = vpux::getHash(seed, get<Index>(val));
        hashItems<Index + 1>(val, seed);
    }
};

}  // namespace std
