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

SymbolSection::SymbolSection(const std::string& name, StringSection* namesSection) : Section(name), m_namesSection(namesSection) {
    m_header.sh_type = SHT_SYMTAB;
    m_header.sh_entsize = sizeof(SymbolEntry);
    m_fileAlignRequirement = alignof(SymbolEntry);

    m_symbols.push_back(std::unique_ptr<Symbol>(new Symbol));
    m_symbols.back()->setIndex(m_symbols.size() - 1);
}

Symbol* SymbolSection::addSymbolEntry(const std::string& name) {
    m_symbols.push_back(std::unique_ptr<Symbol>(new Symbol(name)));
    m_symbols.back()->setIndex(m_symbols.size() - 1);
    return m_symbols.back().get();
}

const std::vector<std::unique_ptr<Symbol>>& SymbolSection::getSymbols() const {
    return m_symbols;
}

void SymbolSection::finalize() {
    std::sort(m_symbols.begin(), m_symbols.end(), [](const std::unique_ptr<Symbol>& lhs, const std::unique_ptr<Symbol>& rhs) {
        return lhs->getBinding() < rhs->getBinding();
    });

    while(m_symbols[m_header.sh_info++]->getBinding() == STB_GLOBAL);

    m_header.sh_link = m_namesSection->getIndex();

    for (const auto& symbol : m_symbols) {
        auto symbolEntry = symbol->m_symbol;
        symbolEntry.st_name = m_namesSection->addString(symbol->getName());
        symbolEntry.st_shndx = symbol->getRelatedSection() ? symbol->getRelatedSection()->getIndex() : 0;

        m_data.insert(m_data.end(), reinterpret_cast<uint8_t*>(&symbolEntry),
                      reinterpret_cast<uint8_t*>(&symbolEntry) + sizeof(symbolEntry));
    }
}
