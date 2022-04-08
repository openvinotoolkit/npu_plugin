//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::PutOpInSectionOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto inputArgDefOp = inputArg().getDefiningOp();
    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_UNLESS(binaryIface != nullptr, "inputArgDefOp is expected to define elf::BinaryOpInterface");

    binaryIface.serialize(binDataSection);
}

size_t vpux::ELF::PutOpInSectionOp::getBinarySize() {
    auto inputArgDefOp = inputArg().getDefiningOp();
    auto binaryIface = llvm::dyn_cast_or_null<vpux::ELF::BinaryOpInterface>(inputArgDefOp);

    VPUX_THROW_UNLESS(binaryIface != nullptr, "inputArgDefOp is expected to define elf::BinaryOpInterface");

    return binaryIface.getBinarySize();
}
