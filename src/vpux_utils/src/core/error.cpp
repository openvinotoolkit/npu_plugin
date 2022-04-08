//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/error.hpp"

#include "vpux/utils/core/logger.hpp"

#include "ie_common.h"

using namespace vpux;

//
// Exceptions
//

[[noreturn]] void vpux::details::throwFormat(StringRef file, int line, const std::string& message) {
    VPUX_UNUSED(file);
    VPUX_UNUSED(line);

#ifdef VPUX_DEVELOPER_BUILD
    Logger::global().error("Got exception in {0}:{1} : {2}", file, line, message);
#else
    Logger::global().error("Got exception : {0}", message);
#endif

    std::stringstream strm;
    strm
#ifndef NDEBUG
            << '\n'
            << file.str() << ':' << line << ' '
#endif
            << message;
    throw InferenceEngine::GeneralError{strm.str()};
}
