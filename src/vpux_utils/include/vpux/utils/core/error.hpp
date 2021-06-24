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

//
// Error handling - helper macros to throw exceptions.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

//
// Exceptions
//

namespace details {

[[noreturn]] void throwFormat(StringRef file, int line, std::string message);

}  // namespace details

#define VPUX_THROW(...) vpux::details::throwFormat(__FILE__, __LINE__, llvm::formatv(__VA_ARGS__).str())

#define VPUX_THROW_UNLESS(_condition_, ...) \
    if (!(_condition_))                     \
    VPUX_THROW(__VA_ARGS__)

#define VPUX_THROW_WHEN(_condition_, ...) \
    if (_condition_)                      \
    VPUX_THROW(__VA_ARGS__)

}  // namespace vpux
