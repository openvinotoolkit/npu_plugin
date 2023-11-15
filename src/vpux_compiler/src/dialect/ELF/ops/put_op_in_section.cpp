//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

void vpux::ELF::PutOpInSectionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto inputArgDefOp = getInputArg().getDefiningOp();

    if (llvm::dyn_cast<vpux::VPURT::DeclareBufferOp>(inputArgDefOp) != nullptr) {
        return;
    }

    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_UNLESS(binaryIface != nullptr,
                      "PutOpInSectionOp::serialize(): inputArg is expected to define elf::BinaryOpInterface");

    binaryIface.serialize(binDataSection);
}

size_t vpux::ELF::PutOpInSectionOp::getBinarySize() {
    auto inputArgDefOp = getInputArg().getDefiningOp();

    if (llvm::dyn_cast<vpux::VPURT::DeclareBufferOp>(inputArgDefOp) != nullptr) {
        // VPURT::DeclareBufferOp is a simple logical operation with no data to serialize.
        auto ndTypeInterface = getInputArg().getType().cast<vpux::NDTypeInterface>();
        return ndTypeInterface.getTotalAllocSize().count();
    }

    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_UNLESS(binaryIface != nullptr,
                      "PutOpInSectionOp::getBinarySize(): inputArg is expected to define elf::BinaryOpInterface");

    return binaryIface.getBinarySize();
}

vpux::VPURT::BufferSection vpux::ELF::PutOpInSectionOp::getMemorySpace() {
    mlir::Value op = getInputArg();

    vpux::ELF::BinaryOpInterface binaryOp = mlir::dyn_cast<vpux::ELF::BinaryOpInterface>(op.getDefiningOp());
    VPUX_THROW_UNLESS(binaryOp != nullptr,
                      "PutOpInSectionOp::getMemorySpace(): inputArg is expected to define elf::BinaryOpInterface");

    return binaryOp.getMemorySpace();
}

vpux::ELF::SectionFlagsAttr vpux::ELF::PutOpInSectionOp::getAccessingProcs() {
    auto inputArgDefOp = getInputArg().getDefiningOp();
    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    // TODO: Add VPUX_THROW_UNLESS if the op does not implement the BinaryOpInterface once relevant VPURT ops are
    // updated

    if (binaryIface) {
        return binaryIface.getAccessingProcs();
    } else {
        return vpux::ELF::SectionFlagsAttr::SHF_NONE;
    }
}

vpux::ELF::SectionFlagsAttr vpux::ELF::PutOpInSectionOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

size_t vpux::ELF::PutOpInSectionOp::getAlignmentRequirements() {
    return ELF::VPUX_NO_ALIGNMENT;
}
