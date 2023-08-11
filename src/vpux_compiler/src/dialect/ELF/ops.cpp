//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/utils/stl_extras.hpp"

//
// initialize
//

void vpux::ELF::ELFDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/ELF/generated/types.cpp.inc>
#include <vpux/compiler/dialect/VPUMI37XX/generated/types.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/dialect.cpp.inc>

#include <vpux/compiler/dialect/ELF/generated/ops_interfaces.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>
