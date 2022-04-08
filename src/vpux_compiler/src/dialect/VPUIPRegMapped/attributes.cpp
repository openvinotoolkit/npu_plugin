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

mlir::Attribute getVPUIPRegMapped_RegisterFieldAttr(::mlir::MLIRContext* context,
                                                    vpux::VPUIPRegMapped::RegFieldType value) {
    return vpux::VPUIPRegMapped::RegisterFieldAttr::get(context, value);
}

mlir::ArrayAttr getVPUIPRegMapped_RegisterFieldArrayAttr(mlir::OpBuilder builder,
                                                         mlir::ArrayRef<vpux::VPUIPRegMapped::RegFieldType> values) {
    auto attrs =
            llvm::to_vector<8>(llvm::map_range(values, [&](vpux::VPUIPRegMapped::RegFieldType v) -> mlir::Attribute {
                return getVPUIPRegMapped_RegisterFieldAttr(builder.getContext(), v);
            }));
    return builder.getArrayAttr(attrs);
}

mlir::Attribute getVPUIPRegMapped_RegisterAttr(::mlir::MLIRContext* context, vpux::VPUIPRegMapped::RegisterType value) {
    return vpux::VPUIPRegMapped::RegisterAttr::get(context, value);
}

mlir::ArrayAttr getVPUIPRegMapped_RegisterArrayAttr(mlir::OpBuilder builder,
                                                    mlir::ArrayRef<vpux::VPUIPRegMapped::RegisterType> values) {
    auto attrs =
            llvm::to_vector<8>(llvm::map_range(values, [&](vpux::VPUIPRegMapped::RegisterType v) -> mlir::Attribute {
                return getVPUIPRegMapped_RegisterAttr(builder.getContext(), v);
            }));
    return builder.getArrayAttr(attrs);
}

//
// RegisterFieldAttr::print() and ::parse() methods
//
void vpux::VPUIPRegMapped::RegisterFieldAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printType(getRegField());
    printer << ">";
}

mlir::Attribute vpux::VPUIPRegMapped::RegisterFieldAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    vpux::VPUIPRegMapped::RegFieldType elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<vpux::VPUIPRegMapped::RegisterFieldAttr>(parser.getContext(), elemType);
}

//
// RegisterFieldAttr::verify
//

mlir::LogicalResult vpux::VPUIPRegMapped::RegisterFieldAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, vpux::VPUIPRegMapped::RegFieldType regField) {
    if (regField == nullptr) {
        return printTo(emitError(), "Got NULL 'regField' in 'RegisterFieldAttr'");
    }

    if (mlir::failed(VPUIPRegMapped::RegFieldType::verify(emitError, regField.getWidth(), regField.getPos(),
                                                          regField.getValue(), regField.getName()))) {
        return printTo(emitError(), "RegisterFieldAttr - invalid.");
    }

    return mlir::success();
}

//
// RegisterAttr::print() and ::parse() methods
//
void vpux::VPUIPRegMapped::RegisterAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printType(getReg());
    printer << ">";
}

mlir::Attribute vpux::VPUIPRegMapped::RegisterAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    vpux::VPUIPRegMapped::RegisterType elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<vpux::VPUIPRegMapped::RegisterAttr>(parser.getContext(), elemType);
}

//
// RegisterAttr::verify
//

mlir::LogicalResult vpux::VPUIPRegMapped::RegisterAttr::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, vpux::VPUIPRegMapped::RegisterType reg) {
    if (reg == nullptr) {
        return printTo(emitError(), "Got NULL 'regField' in 'RegisterFieldAttr'");
    }

    if (mlir::failed(VPUIPRegMapped::RegisterType::verify(emitError, reg.getSize(), reg.getName(), reg.getAddress(),
                                                          reg.getRegFields()))) {
        return printTo(emitError(), "RegisterAttr - invalid.");
    }

    return mlir::success();
}

//
// RegisterMappedAttr::print() and ::parse() methods
//

void vpux::VPUIPRegMapped::RegisterMappedAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printType(getRegMapped());
    printer << ">";
}

mlir::Attribute vpux::VPUIPRegMapped::RegisterMappedAttr::parse(mlir::AsmParser& parser, mlir::Type) {
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
