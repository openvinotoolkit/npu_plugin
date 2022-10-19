//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

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
    auto relocAddend = addend();

    int64_t relocOffset = 0;

    mlir::Operation* op = baseOp().getDefiningOp();

    if (op->hasTrait<vpux::ELF::GetOffsetOfOpInterface::Trait>()) {
        auto opRes0 = op->getResult(0);
        mlir::Type opRes0IndexType = opRes0.getType();
        uint32_t opRes0IndexTypeValue = opRes0IndexType.dyn_cast<vpux::VPUIPRegMapped::IndexType>().getValue();

        if (op->hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
            auto binOpIntf = llvm::cast<vpux::ELF::BinaryOpInterface>(*op);
            relocOffset = opRes0IndexTypeValue * binOpIntf.getBinarySize();
        }

        mlir::Value offsetOf = this->offsetOf();

        auto getOffsetOfOp = llvm::cast<vpux::ELF::GetOffsetOfOpInterface>(*op);
        auto getOffsetOfWithinOperation = getOffsetOfOp.getOffsetOfWithinOperation(offsetOf);
        VPUX_THROW_WHEN(mlir::failed(getOffsetOfWithinOperation),
                        "getOffsetOfWithinOperation failed to retrieve the offset");
        relocOffset += getOffsetOfWithinOperation.getValue();
    }

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(relocOffset);
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

    auto relocType = relocationType();
    auto relocOffset = offset();
    auto relocAddend = addend();

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(relocOffset);
    relocation->setAddend(relocAddend);
}

void vpux::ELF::RelocImmOffsetOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap,
                                            mlir::Operation* targetSection) {
    auto baseOpVal = baseOp();
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

    auto targetSectionOp = llvm::dyn_cast<vpux::ELF::CreateSectionOp>(targetSection);

    auto opIndex = baseOpVal.getType().cast<vpux::VPUIPRegMapped::IndexType>();
    auto opIndexValue = opIndex.getValue();

    size_t totalOffset = 0;

    for (auto& op : targetSectionOp.getOps()) {
        vpux::VPUIPRegMapped::IndexType currOpIndex;
        mlir::Operation* binaryOp = nullptr;
        if (auto putOpInSec = llvm::dyn_cast<vpux::ELF::PutOpInSectionOp>(op)) {
            binaryOp = putOpInSec.inputArg().getDefiningOp();
            currOpIndex = binaryOp->getResult(0).getType().cast<vpux::VPUIPRegMapped::IndexType>();

        } else {
            currOpIndex = op.getResults().front().getType().cast<vpux::VPUIPRegMapped::IndexType>();
            binaryOp = &op;
        }
        if (opIndexValue == currOpIndex.getValue()) {
            break;
        } else {
            if (binaryOp->hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
                auto binOpIntf = llvm::cast<vpux::ELF::BinaryOpInterface>(binaryOp);
                totalOffset += binOpIntf.getBinarySize();
            }
        }
    }

    auto relocType = relocationType();
    auto relocOffset = offset();
    auto relocAddend = addend();

    totalOffset += relocOffset;

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(totalOffset);
    relocation->setAddend(relocAddend);
}
