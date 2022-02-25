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

#include <vpux_elf/writer/relocation_section.hpp>

using namespace elf;
using namespace elf::writer;

RelocationSection::RelocationSection() {
    m_header.sh_type = SHT_RELA;
    m_header.sh_entsize = sizeof(RelocationAEntry);
    m_fileAlignRequirement = alignof(RelocationAEntry);
}

const SymbolSection* RelocationSection::getSymbolTable() const {
    return m_symTab;
}

void RelocationSection::setSymbolTable(const SymbolSection* symTab) {
    m_symTab = symTab;
}

Elf_Word RelocationSection::getSpecialSymbolTable() const {
    return m_header.sh_link;
}

void RelocationSection::setSpecialSymbolTable(Elf_Word specialSymbolTable) {
    m_header.sh_link = specialSymbolTable;
}

const Section* RelocationSection::getSectionToPatch() const {
    return m_sectionToPatch;
}

void RelocationSection::setSectionToPatch(const Section* sectionToPatch) {
    m_sectionToPatch = sectionToPatch;
}

Relocation* RelocationSection::addRelocationEntry() {
    m_relocations.push_back(std::unique_ptr<Relocation>(new Relocation));
    return m_relocations.back().get();
}

const std::vector<Relocation::Ptr>& RelocationSection::getRelocations() const {
    return m_relocations;
}

void RelocationSection::finalize() {
    m_header.sh_info = m_sectionToPatch->getIndex();
    maskFlags(SHF_INFO_LINK);
    if (m_symTab) {
        m_header.sh_link = m_symTab->getIndex();
    }

    for (const auto& relocation : m_relocations) {
        auto relocationEntry = relocation->m_relocation;
        if (relocation->getSymbol()) {
            relocationEntry.r_info = elf64RInfo(relocation->getSymbol()->getIndex(), relocation->getType());
        }

        m_data.insert(m_data.end(), reinterpret_cast<char*>(&relocationEntry),
                      reinterpret_cast<char*>(&relocationEntry) + sizeof(relocationEntry));
    }
}
