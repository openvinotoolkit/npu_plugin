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

#include <elf/writer/symbol_section.hpp>

#include <algorithm>

using namespace elf;
using namespace elf::writer;

//
// SymbolEntry
//

Symbol::Symbol() {
    m_symbol.st_name = 0;
    m_symbol.st_info = elf64STInfo(STB_LOCAL, STT_NOTYPE);
    m_symbol.st_other = elf64STVisibility(STV_DEFAULT);
    m_symbol.st_shndx = 0;
    m_symbol.st_value = 0;
    m_symbol.st_size = 0;
}

std::string Symbol::getName() const {
    return m_name;
}

void Symbol::setName(std::string name) {
    m_name = name;
}

size_t Symbol::getSize() const {
    return m_symbol.st_size;
}

void Symbol::setSize(size_t size) {
    m_symbol.st_size = size;
}

Elf64_Addr Symbol::getValue() const {
    return m_symbol.st_value;
}

void Symbol::setValue(Elf64_Addr value) {
    m_symbol.st_value = value;
}

Elf_Word Symbol::getType() const {
    return elf64STType(m_symbol.st_info);
}

void Symbol::setType(Elf_Word type) {
    m_symbol.st_info = elf64STInfo(elf64STBind(m_symbol.st_info), type);
}

Elf_Word Symbol::getBinding() const {
    return elf64STBind(m_symbol.st_info);
}

void Symbol::setBinding(Elf_Word bind) {
    m_symbol.st_info = elf64STInfo(bind, elf64STType(m_symbol.st_info));
}

uint8_t Symbol::getVisibility() const {
    return m_symbol.st_other;
}

void Symbol::setVisibility(uint8_t visibility) {
    m_symbol.st_other = elf64STVisibility(visibility);
}

const Section* Symbol::getRelatedSection() const {
    return m_relatedSection;
}

void Symbol::setRelatedSection(const Section* section) {
    m_relatedSection = section;
}

size_t Symbol::getIndex() const {
    return m_index;
}

void Symbol::setIndex(size_t index) {
    m_index = index;
}

//
// SymbolSection
//

SymbolSection::SymbolSection(StringSection* namesSection) : m_namesSection(namesSection) {
    m_header.sh_type = SHT_SYMTAB;
    m_header.sh_entsize = sizeof(SymbolEntry);
    m_fileAlignRequirement = alignof(SymbolEntry);

    m_symbols.push_back(std::unique_ptr<Symbol>(new Symbol));
    m_symbols.back()->setIndex(m_symbols.size() - 1);
}

Symbol* SymbolSection::addSymbolEntry() {
    m_symbols.push_back(std::unique_ptr<Symbol>(new Symbol));
    m_symbols.back()->setIndex(m_symbols.size() - 1);
    return m_symbols.back().get();
}

const std::vector<Symbol::Ptr>& SymbolSection::getSymbols() const {
    return m_symbols;
}

void SymbolSection::finalize() {
    std::sort(m_symbols.begin(), m_symbols.end(), [](const Symbol::Ptr& lhs, const Symbol::Ptr& rhs) {
        return lhs->getBinding() < rhs->getBinding();
    });

    m_header.sh_info = std::count_if(m_symbols.cbegin(), m_symbols.cend(), [](const Symbol::Ptr& symbol) {
        return symbol->getBinding() == STB_LOCAL;
    });

    m_header.sh_link = m_namesSection->getIndex();

    for (const auto& symbol : m_symbols) {
        auto symbolEntry = symbol->m_symbol;
        symbolEntry.st_name = m_namesSection->addString(symbol->getName());
        symbolEntry.st_shndx = symbol->getRelatedSection() ? symbol->getRelatedSection()->getIndex() : 0;

        m_data.insert(m_data.end(), reinterpret_cast<char*>(&symbolEntry),
                      reinterpret_cast<char*>(&symbolEntry) + sizeof(symbolEntry));
    }
}
