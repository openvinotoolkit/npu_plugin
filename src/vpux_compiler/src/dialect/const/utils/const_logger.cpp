//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/const_logger.hpp"

//
// Const::logger
//

vpux::Logger& vpux::Const::logger() {
#ifdef VPUX_DEVELOPER_BUILD
    static Logger log("const", LogLevel::Warning);
#else
    static Logger log("const", LogLevel::None);
#endif

    return log;
}
