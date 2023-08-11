//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

void vpux::ELF::CreateLogicalSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                  vpux::ELF::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = secName().str();
    auto section = writer.addEmptySection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));
    section->setAddrAlign(secAddrAlign());

    size_t maxOffset = 0;
    auto block = getBody();
    auto ops = block->getOps<ELF::PutOpInSectionOp>();

    size_t nonBufferAllocs = 0;

    for (auto op : ops) {
        auto pointingOp = op.inputArg().getDefiningOp();

        auto binOp = mlir::dyn_cast<ELF::BinaryOpInterface>(pointingOp);
        VPUX_THROW_UNLESS(binOp, "Operation included in section is not a BinaryOp");

        if (auto bufferOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(pointingOp)) {
            auto currentOffset = bufferOp.byteOffset() + bufferOp.getBinarySize();
            if (currentOffset > maxOffset) {
                maxOffset = currentOffset;
            }
        } else {
            nonBufferAllocs += binOp.getBinarySize();
        }
    }

    section->setSize(maxOffset + nonBufferAllocs);

    sectionMap[getOperation()] = section;
}
