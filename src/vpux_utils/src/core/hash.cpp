//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/hash.hpp"

#include <llvm/ADT/Hashing.h>

using namespace vpux;

size_t vpux::details::combineHashVals(size_t seed, size_t val) {
    return llvm::hash_combine(seed, val);
}

size_t std::hash<StringRef>::operator()(StringRef str) const {
    return llvm::hash_value(str);
}

size_t std::hash<StringLiteral>::operator()(StringLiteral str) const {
    return llvm::hash_value(str);
}
