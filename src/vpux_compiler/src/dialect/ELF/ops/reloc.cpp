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

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::RelocOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap) {
    auto symbolOp = sourceSymbol().getDefiningOp();
    auto actualSymbolOp = llvm::dyn_cast<vpux::ELF::SymbolOp>(symbolOp);

    if (actualSymbolOp.isBuiltin()) {
        auto symInputValue = actualSymbolOp.inputArg();
        auto const_val = llvm::dyn_cast<mlir::arith::ConstantOp>(symInputValue.getDefiningOp());
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
