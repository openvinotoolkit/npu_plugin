//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/exceptions.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <stdarg.h>
#include <sstream>

using namespace vpux;

//
// Exceptions
//

[[noreturn]] void vpux::details::throwNonFormated(const char* file, int line, const std::string& str) {
    VPUX_UNUSED(file);
    VPUX_UNUSED(line);

    /*
    TODO `exception.cpp` is assumed to be in `core` utils package but
    to be fully backward-compatible with `error.cpp`, which depends on LLVM, it
    requires `Logger`. Logger also required LLVM, thsu it is not possible to turn it here

    #ifdef VPUX_DEVELOPER_BUILD
        Logger::global().error("Got exception in {0}:{1} : {2}", file, line, message);
    #else
        Logger::global().error("Got exception : {0}", message);
    #endif
    */
    std::stringstream strm;
    strm
#ifndef NDEBUG
            << '\n'
            << file << ':' << line << ' '
#endif
            << str;
    throw CoreException(strm.str());
}
