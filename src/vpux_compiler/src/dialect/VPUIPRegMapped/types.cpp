//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Debug.h>

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

mlir::LogicalResult vpux::VPUIPRegMapped::RegFieldType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint8_t width, uint8_t pos, uint64_t value,
        std::string name) {
    if (value != 0 && (width < checked_cast<uint8_t>(std::ceil(log2(value + 1))))) {
        return printTo(emitError(),
                       "RegFieldType - provided width {0} is not enough to store provided value {1} for field {2}",
                       width, value, name);
    }
    if (width == 0 || width > Byte(sizeof(value)).to<Bit>().count()) {
        return printTo(emitError(), "RegFieldType - not supported width {0} for field {1}", width, name);
    }
    if (pos + width > Byte(sizeof(value)).to<Bit>().count()) {
        return printTo(emitError(), "RegFieldType - position of start {0} + width {1} Out of Range for field {2}.", pos,
                       width, name);
    }
    if (name.empty()) {
        return printTo(emitError(), "RegFieldType - name is empty.");
    }
    return mlir::success();
}

Byte vpux::VPUIPRegMapped::RegisterType::getSizeInBytes() const {
    return Bit(getSize()).to<Byte>();
}
std::vector<uint8_t> vpux::VPUIPRegMapped::RegisterType::serialize() const {
    std::vector<uint8_t> result(getSizeInBytes().count(), 0);

    uint64_t serializedReg = 0;
    auto fieldsAttrs = getRegFields().getValue();
    for (const auto& fieldAttr : fieldsAttrs) {
        auto pos = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getPos();
        auto value = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getValue();
        auto currentFieldMap = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getMap();

        auto shiftedValue = value << pos;
        serializedReg |= (shiftedValue & currentFieldMap);

        // value and currentFieldMap has max allowed size - 64 bit
        // result should contain first getSize() bytes only
    }
    auto dataPtr = result.data();
    memcpy(dataPtr, &serializedReg, getSizeInBytes().count());
    return result;
}

mlir::LogicalResult vpux::VPUIPRegMapped::RegisterType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint32_t size, std::string name, uint32_t address,
        ::mlir::ArrayAttr regFields) {
    if (name.empty()) {
        return printTo(emitError(), "RegisterType - name is empty.");
    }

    if (size % CHAR_BIT != 0) {
        return printTo(emitError(), "RegisterType - size {0} is not multiple of 8. Allowed sizes: 8, 16, 32, 64", size);
    }

    uint32_t totalWidth(0);
    uint32_t currentAddress(0x0);
    uint64_t wholeRegisterMap(0x0);
    auto fieldsAttrs = regFields.getValue();
    for (const auto& fieldAttr : fieldsAttrs) {
        auto width = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getWidth();
        auto pos = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getPos();
        auto value = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getValue();
        auto name = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getName();
        auto currentFieldMap = fieldAttr.cast<VPUIPRegMapped::RegisterFieldAttr>().getRegField().getMap();
        totalWidth += width;

        // check overlaping
        if (wholeRegisterMap & currentFieldMap) {
            return printTo(emitError(),
                           "RegisterType - Overlap detected. position of start {0} and width {1} for {2} are invalid",
                           pos, width, name);
        }
        wholeRegisterMap |= currentFieldMap;

        if (mlir::failed(vpux::VPUIPRegMapped::RegFieldType::verify(emitError, width, pos, value, name))) {
            return printTo(emitError(), "RegisterType - invalid.");
        }

        if (address < currentAddress) {
            return printTo(emitError(), "RegisterType - address {0} for {1} is invalid", address, name);
        }
        currentAddress = address;
    }

    if (totalWidth > size) {
        return printTo(emitError(), "RegisterType - {0} - invalid size.", totalWidth);
    }

    return mlir::success();
}

std::vector<uint8_t> vpux::VPUIPRegMapped::RegMappedType::serialize() const {
    auto regAttrs = getRegs().getValue();
    std::vector<uint8_t> result(getWidth().count(), 0);
    std::for_each(regAttrs.begin(), regAttrs.end(), [&result](const mlir::Attribute& regAttr) {
        auto reg = regAttr.cast<VPUIPRegMapped::RegisterAttr>().getReg();
        auto serializedRegister = reg.serialize();

        auto resultIter = result.begin() + Byte(reg.getAddress()).count();
        for (auto serializedRegIter = serializedRegister.begin(); serializedRegIter != serializedRegister.end();
             ++serializedRegIter, ++resultIter) {
            *resultIter |= *serializedRegIter;
        }
    });

    return result;
}

Byte vpux::VPUIPRegMapped::RegMappedType::getWidth() const {
    auto regAttrs = getRegs().getValue();
    Byte regMappedSize(0);
    for (auto regAttr = regAttrs.begin(); regAttr != regAttrs.end(); regAttr++) {
        auto reg = regAttr->cast<VPUIPRegMapped::RegisterAttr>().getReg();
        auto boundingRegMappedWidth = Byte(reg.getAddress()) + reg.getSizeInBytes();
        regMappedSize = boundingRegMappedWidth > regMappedSize ? boundingRegMappedWidth : regMappedSize;
    }
    return regMappedSize;
}

mlir::LogicalResult vpux::VPUIPRegMapped::RegMappedType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, std::string name, ::mlir::ArrayAttr regs) {
    if (name.empty()) {
        return printTo(emitError(), "RegMappedType - name is empty.");
    }

    auto regAttrs = regs.getValue();
    for (const auto& regAttr : regAttrs) {
        auto reg = regAttr.cast<VPUIPRegMapped::RegisterAttr>().getReg();
        auto regSize = reg.getSize();
        auto name = reg.getName();
        auto address = reg.getAddress();
        auto regFields = reg.getRegFields();
        if (mlir::failed(vpux::VPUIPRegMapped::RegisterType::verify(emitError, regSize, name, address, regFields))) {
            return printTo(emitError(), "RegMappedType {0} - invalid.", name);
        }
    }

    return mlir::success();
}
