//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include <elf/writer.hpp>
#include "vpux/compiler/utils/stl_extras.hpp"

//
// initialize
//

void vpux::ELF::ELFDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/ELF/generated/types.cpp.inc>
            >();
}

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/dialect.cpp.inc>

#include <vpux/compiler/dialect/ELF/generated/ops_interfaces.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/ELF/generated/ops.cpp.inc>

namespace {
using SectionMap = vpux::OpOrderedMap<elf::writer::Section*>;
//TODO::this works only if IR is being immutable troughout the life-time of the map....
//which in our case it is.... don't we have nicer way of mapping mlir::Operation's????
using SymbolMap = std::map<mlir::Operation*, elf::writer::Symbol*>;

mlir::Operation* opParentSection(mlir::Value val) {

    //If one of the users of the value is a "PutAnyOpInSectionOp", then we are interested in the encapsulation section of it.
    //Based on IR validity, assume, any "PutAnyOpInSectionOp" is always in a "Section" op
    //If it has no successors of this type, then based on IR validity we assume that the op itself is placed in a section
    mlir::Operation* op = nullptr;
    for(mlir::Operation* user : val.getUsers()) {
        if(auto placeholder = llvm::dyn_cast_or_null<vpux::ELF::PutAnyOpInSectionOp>(user)) {
            op = user;
            break;
        }
    }
    if(op == nullptr) {
        op = val.getDefiningOp();
    }

    auto region = op->getParentRegion();
    return region->getParentOp();
}

}

void vpux::ELF::CreateSectionOp::serialize(elf::Writer& writer, SectionMap& sectionMap, SymbolMap& symbolMap) {
    const auto name = secName().str();
    auto section = writer.addBinaryDataSection<char>(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));
    section->setAddrAlign(secAddrAlign());

    auto block = getBody();
    for(auto& op : block->getOperations()) {
        if(op.hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELF::BinaryOpInterface>(op);

            std::vector<char> buf;
            binaryOp.serialize(buf);
            section->appendData(buf.data(), buf.size());
        }
    }

    sectionMap[getOperation()] = section;
    VPUX_UNUSED(symbolMap);

    return;
}

void vpux::ELF::CreateSymbolTableSectionOp::serialize(elf::Writer& writer, SectionMap& sectionMap, SymbolMap& symbolMap) {
    const auto name = secName().str();
    auto section = writer.addSymbolSection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));

    auto block = getBody();
    for(auto& op : block->getOperations()) {
        elf::writer::Symbol* symbol = section->addSymbolEntry();

        if(vpux::ELF::SymbolOp symOp = mlir::dyn_cast<vpux::ELF::SymbolOp>(op)) {
            symOp.serialize(symbol,sectionMap);
            symbolMap[symOp.getOperation()] = symbol;
        }
        else if(vpux::ELF::PutAnyOpInSectionOp placeholder = mlir::dyn_cast<vpux::ELF::PutAnyOpInSectionOp>(op)) {
            auto actualOp = placeholder.inputArg().getDefiningOp();
            vpux::ELF::SymbolOp symOp = mlir::dyn_cast<vpux::ELF::SymbolOp>(actualOp);

            VPUX_THROW_UNLESS(symOp, "Symbol table section op is expected to have placeholders that point to SymbolOps only");

            symOp.serialize(symbol,sectionMap);
            symbolMap[symOp.getOperation()] = symbol;
        }
        else {
            VPUX_THROW("Symbol table section op is expected to have either SymbolOps or placeholders that point to SymbolOps");
        }
    }

    sectionMap[getOperation()] = section;

    return;
}

void vpux::ELF::CreateRelocationSectionOp::serialize(elf::Writer& writer, SectionMap& sectionMap, SymbolMap& symbolMap) {
    const auto name = secName().str();
    auto section = writer.addRelocationSection(name);

    //Look up dependent sections
    auto symTab = mlir::dyn_cast<vpux::ELF::CreateSymbolTableSectionOp>(sourceSymbolTableSection().getDefiningOp());
    VPUX_THROW_UNLESS(symTab, "Reloc section expected to point at a symbol table section");

    auto symTabMapEntry = sectionMap.find(symTab.getOperation());
    VPUX_THROW_UNLESS(symTabMapEntry != sectionMap.end(), "Serializing a reloc section before serializing it's dependent SymTabSection");

    elf::writer::Section* symTabSection = symTabMapEntry->second;

    auto target = mlir::dyn_cast<vpux::ELF::CreateSectionOp>(targetSection().getDefiningOp());
    VPUX_THROW_UNLESS(target, "Reloc section expected to point ot a valid Section op");

    auto targetMapEntry = sectionMap.find(target.getOperation());
    VPUX_THROW_UNLESS(targetMapEntry != sectionMap.end(), "Serializing a reloc section before serializing it's dependent target section");

    elf::writer::Section* targetSection = targetMapEntry->second;

    section->setSymbolTable(dynamic_cast<elf::writer::SymbolSection*>(symTabSection));
    section->setSectionToPatch(targetSection);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));
    section->maskFlags(elf::SHF_INFO_LINK); //TODO(Review) : @abakalin , is there any scenario where we would not need to set this link? shoould we expose the option in the dialect?

    auto block = getBody();
    for(auto& op : block->getOperations()) {
        elf::writer::Relocation* relocation = section->addRelocationEntry();

        if(vpux::ELF::RelocOp relocOp = mlir::dyn_cast<vpux::ELF::RelocOp>(op)) {
            relocOp.serialize(relocation, symbolMap);
        }
        else if(vpux::ELF::PutAnyOpInSectionOp placeholder = mlir::dyn_cast<vpux::ELF::PutAnyOpInSectionOp>(op)) {
            auto actualOp = placeholder.inputArg().getDefiningOp();
            vpux::ELF::RelocOp relocOp = mlir::dyn_cast<vpux::ELF::RelocOp>(actualOp);

            VPUX_THROW_UNLESS(relocOp, "Relocation section op is expected to have placeholders that point to RelocOps only");

            relocOp.serialize(relocation, symbolMap);
        }
        else {
            VPUX_THROW("Relocation Section op is expected to have either RelocOps or placeholders that point to RelocOps");
        }
    }

    sectionMap[getOperation()] = section;

    return;
}

void vpux::ELF::PutAnyOpInSectionOp::serialize(std::vector<char>& writer) {

    auto parent = inputArg().getDefiningOp();
    auto binaryIface = llvm::dyn_cast<vpux::ELF::BinaryOpInterface>(parent);

    VPUX_THROW_UNLESS(binaryIface != nullptr, "Parent of PutAnyOpInSection is expected to define elf::BinaryOpInterface");

    binaryIface.serialize(writer);

    return;
}

void vpux::ELF::SymbolOp::serialize(elf::writer::Symbol* symbol, SectionMap& sectionMap) {

    auto symName = name().getValueOr("");
    auto symType = type().getValueOr(vpux::ELF::SymbolTypeAttr::STT_NOTYPE);
    auto symSize = size().getValueOr(0);
    auto symVal = value().getValueOr(0);

    /*From serialization perspective symbols can be of 5 types
        - SECTION symbols : in this case the parentSection is the defining opt itself
        - Generic symbols : Symbols representing an OP inside the IR. In this case we need the parent section of either the OP or it's placeholder
        - I/O symbols : symbols that represent function arguments. In this case we will not have a parentSection, and no relatedSection
        - Symbols refering to the "Special Symbol Table" : TODO(iszilve)  handle this case
        - Standalone symbols : symbols that do not relate to any entity inside the IR (nor the ELF itself ). TODO(iszilve) handlle this case
    */

    mlir::Operation* parentSection = nullptr;
    if(inputArg().getDefiningOp()) {
        if(llvm::isa<ELF::ElfSectionInterface>(inputArg().getDefiningOp())) {
            parentSection = inputArg().getDefiningOp();
        }
        else {
            parentSection = opParentSection(inputArg());
            VPUX_THROW_UNLESS(parentSection," Could not find valid parent section for op");
        }
    }

    symbol->setName(symName.str());
    symbol->setType(static_cast<elf::Elf_Word>(symType));
    symbol->setSize(symSize);
    symbol->setValue(symVal);

    if(parentSection){
        auto sectionMapEntry = sectionMap.find(parentSection);
        VPUX_THROW_UNLESS(sectionMapEntry != sectionMap.end(), "Unable to find section entry for SymbolOp");
        auto sectionEntry = sectionMapEntry->second;

        symbol->setRelatedSection(sectionEntry);
    }

    return;
}

void vpux::ELF::RelocOp::serialize(elf::writer::Relocation* relocation, SymbolMap& symbolMap) {

    auto symbolOp = sourceSymbol().getDefiningOp();
    auto symbolMapEntry = symbolMap.find(symbolOp);
    VPUX_THROW_UNLESS(symbolMapEntry != symbolMap.end(), "Unable to locate symbol entry for relocation");
    auto symbolEntry = symbolMapEntry->second;

    auto relocType = relocationType();
    auto relocOffset = offsetTargetField();
    auto relocAddend = addend();

    relocation->setSymbol(symbolEntry);
    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(relocOffset);
    relocation->setAddend(relocAddend);

    return;

}
