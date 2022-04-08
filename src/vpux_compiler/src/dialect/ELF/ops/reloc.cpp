//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::RelocOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap) {
    auto symbolOp = sourceSymbol().getDefiningOp();
    auto actualSymbolOp = llvm::cast<vpux::ELF::SymbolOp>(symbolOp);

    if (actualSymbolOp.isBuiltin()) {
        auto symInputValue = actualSymbolOp.inputArg();
        auto const_val = llvm::cast<mlir::arith::ConstantOp>(symInputValue.getDefiningOp());
        auto symValue = const_val.getValue().cast<mlir::IntegerAttr>().getInt();

        relocation->setSpecialSymbol(static_cast<elf::Elf_Word>(symValue));
    } else {
        auto symbolMapEntry = symbolMap.find(symbolOp);
        VPUX_THROW_UNLESS(symbolMapEntry != symbolMap.end(), "Unable to locate symbol entry for relocation");
        auto symbolEntry = symbolMapEntry->second;
        relocation->setSymbol(symbolEntry);
    }

    auto relocType = relocationType();
    auto relocOffset = offsetTargetField();
    auto relocAddend = addend();

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(relocOffset);
    relocation->setAddend(relocAddend);
}
