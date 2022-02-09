//
// Copyright Intel Corporation.
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
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/dialect.cpp.inc>

#include <vpux/compiler/dialect/ELF/generated/ops_interfaces.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>
