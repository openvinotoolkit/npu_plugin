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

#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "llvm/Support/Debug.h"

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// register Types
//

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/types.cpp.inc>
            >();
}


//
// Dialect hooks
//

mlir::Type vpux::VPUIPRegMapped::VPUIPRegMappedDialect::parseType(mlir::DialectAsmParser& parser) const {
    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get VPUIPRegMapped Type mnemonic");
        return nullptr;
    }

    mlir::Type type;
    if (!generatedTypeParser(parser, mnemonic, type).hasValue()) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Unknown VPUIPRegMapped Type '{0}'", mnemonic);
    }

    return type;
}

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedTypePrinter(type, os)), "Got unsupported Type : {}", type);
}
