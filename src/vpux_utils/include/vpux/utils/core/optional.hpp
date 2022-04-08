//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// `std::optional` analogue.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/hash.hpp"

#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>

namespace vpux {

using llvm::None;
using llvm::Optional;

}  // namespace vpux

//
// std::hash specialization
//

namespace std {

template <typename T>
struct hash<vpux::Optional<T>> final {
    size_t operator()(const vpux::Optional<T>& opt) const {
        return opt.has_value() ? vpux::getHash(opt.value()) : 0;
    }
};

}  // namespace std

//
// llvm::format_provider specialization
//

namespace llvm {

template <typename T>
struct format_provider<Optional<T>> final {
    static void format(const Optional<T>& opt, llvm::raw_ostream& stream, StringRef style) {
        if (opt.hasValue()) {
            llvm::detail::build_format_adapter(opt.getValue()).format(stream, style);
        } else {
            stream << "<NONE>";
        }
    }
};

}  // namespace llvm
