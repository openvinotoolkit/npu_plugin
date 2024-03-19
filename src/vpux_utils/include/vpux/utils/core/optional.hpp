//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// `std::optional` header forwarding and format_provider.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/hash.hpp"

#include <optional>

//
// llvm::format_provider specialization
//

namespace llvm {

template <typename T>
struct format_provider<std::optional<T>> final {
    static void format(const std::optional<T>& opt, llvm::raw_ostream& stream, StringRef style) {
        if (opt.has_value()) {
            llvm::detail::build_format_adapter(opt.value()).format(stream, style);
        } else {
            stream << "<NONE>";
        }
    }
};

}  // namespace llvm
