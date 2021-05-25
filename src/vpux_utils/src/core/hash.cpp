//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
