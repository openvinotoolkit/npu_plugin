//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/types.hpp"

#include "vpux/compiler/dialect/ELF/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Debug.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/ELF/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// ELFDialect::registerTypes
//

void vpux::ELF::ELFDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/ELF/types.cpp.inc>
#include <vpux/compiler/dialect/VPUMI37XX/types.cpp.inc>
            >();
}
