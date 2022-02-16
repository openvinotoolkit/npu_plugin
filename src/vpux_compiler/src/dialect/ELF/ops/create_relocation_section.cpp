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

void vpux::ELF::CreateRelocationSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                     vpux::ELF::SymbolMapType& symbolMap) {
    const auto name = secName().str();
    auto section = writer.addRelocationSection(name);

    // Look up dependent sections
    auto symTab = llvm::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(sourceSymbolTableSection().getDefiningOp());
    VPUX_THROW_UNLESS(symTab != nullptr, "Reloc section expected to refer to a symbol table section");

    auto symTabMapEntry = sectionMap.find(symTab.getOperation());
    VPUX_THROW_UNLESS(symTabMapEntry != sectionMap.end(),
                      "Can't serialize a reloc section that doesn't have its dependent symbol table section");

    auto symTabSection = symTabMapEntry->second;

    auto target = llvm::dyn_cast_or_null<vpux::ELF::CreateSectionOp>(targetSection().getDefiningOp());
    VPUX_THROW_UNLESS(target != nullptr, "Reloc section expected to refer at a valid target section");

    auto targetMapEntry = sectionMap.find(target.getOperation());
    VPUX_THROW_UNLESS(targetMapEntry != sectionMap.end(),
                      "Can't serialize a reloc section that doesn't have its dependent target section");

    auto targetSection = targetMapEntry->second;

    section->setSymbolTable(dynamic_cast<elf::writer::SymbolSection*>(symTabSection));
    section->setSectionToPatch(targetSection);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        auto relocation = section->addRelocationEntry();

        if (auto relocOp = llvm::dyn_cast<vpux::ELF::RelocOp>(op)) {
            relocOp.serialize(relocation, symbolMap);
        } else if (auto placeholder = llvm::dyn_cast<vpux::ELF::PutOpInSectionOp>(op)) {
            auto actualOp = placeholder.inputArg().getDefiningOp();
            auto relocOp = llvm::dyn_cast<vpux::ELF::RelocOp>(actualOp);

            VPUX_THROW_UNLESS(relocOp != nullptr,
                              "CreateRelocationSection is expected to have PutOpInSectionOp that refer to RelocOps "
                              "only. Got *actualOp {0}.",
                              *actualOp);

            relocOp.serialize(relocation, symbolMap);
        } else {
            VPUX_THROW(
                    "CreateRelocationSection op is expected to have either RelocOps or PutOpInSectionOp that refer to "
                    "RelocOps");
        }
    }

    sectionMap[getOperation()] = section;
}
