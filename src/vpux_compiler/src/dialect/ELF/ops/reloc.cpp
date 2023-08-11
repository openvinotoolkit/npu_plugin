//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/types.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

namespace {
mlir::Value getTargetSectionOfRelocOp(mlir::Operation* op) {
    auto parentOp = op->getParentOp();
    auto relocSectionOp = mlir::dyn_cast<ELF::CreateRelocationSectionOp>(parentOp);
    VPUX_THROW_UNLESS(relocSectionOp, "Parent Op of RelocOp must be an ELF::CreateRelocationSectionOp");

    return relocSectionOp.targetSection();
}
}  // namespace

void vpux::ELF::RelocOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap) {
    auto baseOpVal = baseOp();
    auto symbolOp = sourceSymbol().getDefiningOp();
    auto actualSymbolOp = llvm::cast<vpux::ELF::SymbolOp>(symbolOp);
    auto targetSection = getTargetSectionOfRelocOp(getOperation());

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

    mlir::Operation* baseOperation = baseOpVal.getDefiningOp();
    auto relocType = relocationType();
    auto relocAddend = addend();

    auto totalOffset = ELF::getOffsetOfOpInSection(baseOpVal, targetSection);

    auto getOffsetOfOpIf = mlir::dyn_cast<vpux::ELF::GetOffsetOfOpInterface>(baseOperation);
    VPUX_THROW_UNLESS(
            getOffsetOfOpIf,
            "Value given as offsetOf parameter does not represent an Op that implements getOffsetOfOpInterface");
    auto computedOffsetOf = getOffsetOfOpIf.getOffsetOfWithinOperation(offsetOf());

    totalOffset += computedOffsetOf.value_or(0);

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(totalOffset);
    relocation->setAddend(relocAddend);
}

void vpux::ELF::RelocImmOffsetOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap) {
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

    auto baseOpVal = baseOp();
    auto relocType = relocationType();
    auto relocAddend = addend();
    auto relocOffset = offset();

    if (baseOpVal) {
        auto targetSection = getTargetSectionOfRelocOp(getOperation());
        relocOffset += ELF::getOffsetOfOpInSection(baseOpVal, targetSection);
    }

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(relocOffset);
    relocation->setAddend(relocAddend);
}
