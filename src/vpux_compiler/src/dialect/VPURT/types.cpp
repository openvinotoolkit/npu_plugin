//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPURT/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPURTDialect::registerTypes
//

void vpux::VPURT::VPURTDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPURT/generated/types.cpp.inc>
            >();
}
