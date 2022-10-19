//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

void vpux::ELF::PutOpInSectionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto inputArgDefOp = inputArg().getDefiningOp();

    if (llvm::dyn_cast<vpux::VPURT::DeclareBufferOp>(inputArgDefOp) != nullptr) {
        return;
    }

    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_UNLESS(binaryIface != nullptr,
                      "PutOpInSectionOp::serialize(): inputArgDefOp is expected to define elf::BinaryOpInterface");

    binaryIface.serialize(binDataSection);
}

size_t vpux::ELF::PutOpInSectionOp::getBinarySize() {
    auto inputArgDefOp = inputArg().getDefiningOp();

    if (llvm::dyn_cast<vpux::VPURT::DeclareBufferOp>(inputArgDefOp) != nullptr) {
        // VPURT::DeclareBufferOp is a simple logical operation with no data to serialize.
        auto memrefType = inputArg().getType().cast<mlir::MemRefType>();
        return memrefType.getSizeInBits() / CHAR_BIT;
    }

    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_UNLESS(binaryIface != nullptr,
                      "PutOpInSectionOp::getBinarySize(): inputArgDefOp is expected to define elf::BinaryOpInterface");

    return binaryIface.getBinarySize();
}
