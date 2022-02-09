//
// Copyright 2022 Intel Corporation.
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

#include <elf/writer.hpp>
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
