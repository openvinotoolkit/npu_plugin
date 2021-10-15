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

#include "vpux/compiler/dialect/ELF/types.hpp"

#include "vpux/compiler/dialect/ELF/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "llvm/Support/Debug.h"  // Alex: 2021_10_05

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
    llvm::dbgs() << "Alex: vpux::ELF::ELFDialect::parseType(): Entered the method.\n";

    StringRef mnemonic;
    if (mlir::failed(parser.parseKeyword(&mnemonic))) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Failed to get ELF Type mnemonic");
        return nullptr;
    }

    llvm::dbgs() << "Alex: ELF::ELFDialect::parseType(): mnemonic = " << mnemonic << "\n";
    llvm::dbgs() << "Alex: vpux::ELF::ELFDialect::parseType(): After parser.parseKeyword().\n";

    mlir::Type type;
    if (!generatedTypeParser(getContext(), parser, mnemonic, type).hasValue()) {
        printTo(parser.emitError(parser.getCurrentLocation()), "Unknown ELF Type '{0}'", mnemonic);
    }

    llvm::dbgs() << "Alex: vpux::ELF::ELFDialect::parseType(): After generatedTypeParser().\n";
    // llvm::dbgs() << "Alex: ELF::ELFDialect::parseType(): getContext() = " << getContext() << "\n";
    // <<error: no match for ‘operator<<’ (operand types are ‘llvm::raw_ostream’ and ‘mlir::MLIRContext’)>>:
    // llvm::dbgs() << "Alex: ELF::ELFDialect::parseType(): *getContext() = " << (*getContext()) << "\n";
    //
    llvm::dbgs() << "Alex: ELF::ELFDialect::parseType(): type = " << type << "\n";

    return type;
}

void vpux::ELF::ELFDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& os) const {
    llvm::dbgs() << "Alex: vpux::ELF::ELFDialect::printType(): Entered the method.\n";

    VPUX_THROW_UNLESS(mlir::succeeded(generatedTypePrinter(type, os)), "Got unsupported Type : {}", type);

    llvm::dbgs() << "Alex: vpux::ELF::ELFDialect::printType(): Exiting the method.\n";
}
