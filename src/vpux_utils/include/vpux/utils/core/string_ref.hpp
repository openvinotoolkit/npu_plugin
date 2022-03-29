//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Light-weight non-owning reference to string.
//

#pragma once

#include "vpux/utils/core/hash.hpp"

#include <llvm/ADT/StringRef.h>

namespace vpux {

using llvm::StringLiteral;
using llvm::StringRef;

}  // namespace vpux

//
// std::hash specialization
//

namespace std {

template <>
struct hash<vpux::StringRef> final {
    size_t operator()(vpux::StringRef str) const {
        return llvm::hash_value(str);
    }
};

template <>
struct hash<vpux::StringLiteral> final {
    size_t operator()(vpux::StringLiteral str) const {
        return llvm::hash_value(str);
    }
};

}  // namespace std
