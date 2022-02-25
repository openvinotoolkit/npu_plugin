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

#include <vpux_elf/writer/symbol_section.hpp>

#include <algorithm>

using namespace elf;
using namespace elf::writer;

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
