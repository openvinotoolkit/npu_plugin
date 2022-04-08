//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/types.hpp"

#include "vpux/compiler/dialect/ELF/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "llvm/Support/Debug.h"

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/ELF/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// Dialect hooks
//

mlir::Type vpux::ELF::ELFDialect::parseType(mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get ELF Type mnemonic");
        return nullptr;
    }

    mlir::Type type;
    if (!generatedTypeParser(parser, mnemonic, type).hasValue()) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Unknown ELF Type '{0}'", mnemonic);
    }

    return type;
}

void vpux::ELF::ELFDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedTypePrinter(type, os)), "Got unsupported Type : {}", type);
}
