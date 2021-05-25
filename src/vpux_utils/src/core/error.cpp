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

#include "vpux/utils/core/error.hpp"

#include "vpux/utils/core/logger.hpp"

#include "ie_common.h"

using namespace vpux;

//
// Exceptions
//

[[noreturn]] void vpux::details::throwFormat(StringRef file, int line, std::string message) {
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
