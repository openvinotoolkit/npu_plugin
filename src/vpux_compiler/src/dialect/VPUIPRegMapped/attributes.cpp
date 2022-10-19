//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/attributes.cpp.inc>
#undef GET_ATTRDEF_CLASSES

//
// Dialect hooks
//

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPUIPRegMapped/generated/attributes.cpp.inc>
            >();
}

mlir::Attribute vpux::VPUIPRegMapped::VPUIPRegMappedDialect::parseAttribute(mlir::DialectAsmParser& parser,
                                                                            mlir::Type type) const {
    StringRef mnemonic;
    VPUX_THROW_UNLESS(mlir::succeeded(parser.parseKeyword(&mnemonic)), "Can't get attribute mnemonic");

    mlir::Attribute attr;
    const auto res = generatedAttributeParser(parser, mnemonic, type, attr);
    VPUX_THROW_UNLESS(res.hasValue() && mlir::succeeded(res.getValue()), "Can't parse attribute");

    return attr;
}

void vpux::VPUIPRegMapped::VPUIPRegMappedDialect::printAttribute(mlir::Attribute attr,
                                                                 mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedAttributePrinter(attr, os)), "Got unsupported attribute '{0}'", attr);
}

//
// RegisterMappedAttr::print() and ::parse() methods
//

void vpux::VPUIPRegMapped::RegisterMappedAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printType(getRegMapped());
    printer << ">";
}

mlir::Attribute vpux::VPUIPRegMapped::RegisterMappedAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    vpux::VPUIPRegMapped::RegMappedType elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<vpux::VPUIPRegMapped::RegisterMappedAttr>(parser.getContext(), elemType);
}

//
// RegisterMappedAttr::verify
//

mlir::LogicalResult vpux::VPUIPRegMapped::RegisterMappedAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, vpux::VPUIPRegMapped::RegMappedType regMapped) {
    if (regMapped == nullptr) {
        return printTo(emitError(), "Got NULL 'regMapped' in 'RegisterMappedAttr'");
    }

    if (mlir::failed(VPUIPRegMapped::RegMappedType::verify(emitError, regMapped.getName(), regMapped.getRegs()))) {
        return printTo(emitError(), "RegisterMappedAttr - invalid.");
    }

    return mlir::success();
}
