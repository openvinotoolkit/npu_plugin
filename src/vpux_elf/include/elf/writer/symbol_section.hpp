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

#pragma once

#include <elf/types/symbol_entry.hpp>
#include <elf/writer/section.hpp>

#include <elf/writer/string_section.hpp>

#include <list>

namespace elf {

class Writer;

namespace writer {

class SymbolSection;

class Symbol {
public:
    using Ptr = std::unique_ptr<Symbol>;

public:
    std::string getName() const;
    void setName(std::string name);

    Elf_Word getType() const;
    void setType(Elf_Word type);

    Elf_Word getBinding() const;
    void setBinding(Elf_Word bind);

    uint8_t getVisibility() const;
    void setVisibility(uint8_t visibility);

    const Section* getRelatedSection() const;
    void setRelatedSection(const Section* section);

    Elf64_Addr getValue() const;
    void setValue(Elf64_Addr value);

    size_t getSize() const;
    void setSize(size_t size);

    size_t getIndex() const;

private:
    Symbol();

    void setIndex(size_t index);

private:
    std::string m_name;
    size_t m_index = 0;

    SymbolEntry m_symbol{};
    const Section* m_relatedSection = nullptr;

    friend SymbolSection;
};

class SymbolSection final : public Section {
public:
    Symbol* addSymbolEntry();
    const std::vector<Symbol::Ptr>& getSymbols() const;

private:
    explicit SymbolSection(StringSection* namesSection);

    void finalize() override;

private:
    StringSection* m_namesSection;
    std::vector<Symbol::Ptr> m_symbols;

    friend Writer;
};

} // namespace writer
} // namespace elf
