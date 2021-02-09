//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/utils/core/error.hpp"

#include "vpux/utils/core/logger.hpp"

#include <details/ie_exception.hpp>

using namespace vpux;

//
// Exceptions
//

[[noreturn]] void vpux::details::throwFormat(StringRef file, int line, std::string message) {
    VPUX_UNUSED(file);
    VPUX_UNUSED(line);

#ifdef NDEBUG
    Logger::global().error("Got exception : {0}", message);
#else
    Logger::global().error("Got exception in {0}:{1} : {2}", file, line, message);
#endif

    throw InferenceEngine::details::InferenceEngineException(file.str(), line, message);
}
