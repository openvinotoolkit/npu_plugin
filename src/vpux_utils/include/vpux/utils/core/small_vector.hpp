//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// `std::vector` like container with pre-allocated buffer for small sizes.
//

#pragma once

#include "vpux/utils/core/hash.hpp"

#include <llvm/ADT/SmallVector.h>

namespace vpux {

using llvm::SmallVector;
using llvm::SmallVectorImpl;

}  // namespace vpux

//
// std::hash specialization
//

namespace std {

template <typename T, unsigned N>
struct hash<vpux::SmallVector<T, N>> final {
    size_t operator()(const vpux::SmallVector<T, N>& vec) const {
        return vpux::getRangeHash(vec);
    }
};

}  // namespace std
