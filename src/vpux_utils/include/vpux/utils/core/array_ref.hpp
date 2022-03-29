//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Non-owning wrapper for array range.
//

#pragma once

#include "vpux/utils/core/hash.hpp"

#include <llvm/ADT/ArrayRef.h>

namespace vpux {

using llvm::ArrayRef;
using llvm::MutableArrayRef;

using llvm::makeArrayRef;
using llvm::makeMutableArrayRef;

}  // namespace vpux

//
// std::hash specialization
//

namespace std {

template <typename T>
struct hash<vpux::ArrayRef<T>> final {
    size_t operator()(vpux::ArrayRef<T> arr) const {
        return vpux::getRangeHash(arr);
    }
};

template <typename T>
struct hash<vpux::MutableArrayRef<T>> final {
    size_t operator()(vpux::MutableArrayRef<T> arr) const {
        return vpux::getRangeHash(arr);
    }
};

}  // namespace std
