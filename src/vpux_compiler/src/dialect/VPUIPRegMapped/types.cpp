//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

mlir::LogicalResult vpux::VPUIPRegMapped::RegFieldType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint8_t width, uint8_t pos, uint32_t value,
        std::string name) {
    VPUX_UNUSED(value);
    if (width == 0 || width > 32) {
        return printTo(emitError(), "RegFieldType - not supported width");
    }
    if (pos > 31) {
        return printTo(emitError(), "RegFieldType - position of start Out of Range.");
    }
    if (name.empty()) {
        return printTo(emitError(), "RegFieldType - name is empty.");
    }
    return mlir::success();
}

void vpux::VPUIPRegMapped::RegisterType::print(::mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<" << getImpl()->size << "," << getImpl()->name << ">";
    for (size_t idx = 0; idx < getImpl()->regFields.size(); ++idx) {
        getImpl()->regFields[idx].print(printer);
    }
}

::mlir::Type vpux::VPUIPRegMapped::RegisterType::parse(::mlir::DialectAsmParser& parser) {
    if (parser.parseLess())
        return Type();
    uint32_t size(0);
    std::string name;
    if (parser.parseInteger(size))
        return Type();
    if (parser.parseComma())
        return Type();
    if (parser.parseString(&name))
        return Type();
    if (parser.parseGreater())
        return Type();

    ::mlir::SmallVector<RegFieldType> regFields;
    if (parser.parseCommaSeparatedList(::mlir::OpAsmParser::Delimiter::LessGreater, [&]() -> ::mlir::ParseResult {
            vpux::VPUIPRegMapped::RegFieldType regField;
            if (parser.parseType(regField))
                return ::mlir::failure();
            regFields.push_back(regField);
            return ::mlir::success();
        }))
        return Type();

    return get(parser.getContext(), size, name, regFields);
}

mlir::LogicalResult vpux::VPUIPRegMapped::RegisterType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint32_t size, std::string name,
        ::llvm::ArrayRef<RegFieldType> regFields) {
    if (name.empty()) {
        return printTo(emitError(), "RegisterType - name is empty.");
    }

    uint32_t totalWidth(0);
    for (size_t idx = 0; idx < regFields.size(); ++idx) {
        auto width = regFields[idx].getWidth();
        auto pos = regFields[idx].getPos();
        auto value = regFields[idx].getValue();
        auto name = regFields[idx].getName();
        totalWidth = width;
        if (mlir::failed(vpux::VPUIPRegMapped::RegFieldType::verify(emitError, width, pos, value, name))) {
            return printTo(emitError(), "RegisterType - invalid.");
        }
    }

    if (totalWidth != size) {
        return printTo(emitError(), "RegisterType - invalid size.");
    }

    return mlir::success();
}

void vpux::VPUIPRegMapped::RegMappedType::print(::mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<" << getImpl()->name << ">";
    for (size_t idx = 0; idx < getImpl()->regs.size(); ++idx) {
        getImpl()->regs[idx].print(printer);
    }
}

::mlir::Type vpux::VPUIPRegMapped::RegMappedType::parse(::mlir::DialectAsmParser& parser) {
    if (parser.parseLess())
        return Type();
    std::string name;
    if (parser.parseString(&name))
        return Type();
    if (parser.parseGreater())
        return Type();

    ::mlir::SmallVector<RegisterType> registers;
    if (parser.parseCommaSeparatedList(::mlir::OpAsmParser::Delimiter::LessGreater, [&]() -> ::mlir::ParseResult {
            vpux::VPUIPRegMapped::RegisterType reg;
            if (parser.parseType(reg))
                return ::mlir::failure();
            registers.push_back(reg);
            return ::mlir::success();
        }))
        return Type();

    return get(parser.getContext(), name, registers);
}

mlir::LogicalResult vpux::VPUIPRegMapped::RegMappedType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, std::string name,
        ::llvm::ArrayRef<RegisterType> regs) {
    if (name.empty()) {
        return printTo(emitError(), "RegMappedType - name is empty.");
    }

    for (size_t idx = 0; idx < regs.size(); ++idx) {
        auto regSize = regs[idx].getSize();
        auto name = regs[idx].getName();
        auto regFields = regs[idx].getRegFields();
        if (mlir::failed(vpux::VPUIPRegMapped::RegisterType::verify(emitError, regSize, name, regFields))) {
            return printTo(emitError(), "RegMappedType - invalid.");
        }
    }

    return mlir::success();
}
