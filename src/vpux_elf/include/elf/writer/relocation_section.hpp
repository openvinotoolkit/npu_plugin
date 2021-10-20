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

#include <elf/writer/section.hpp>

#include <elf/types/relocation_entry.hpp>
#include <elf/writer/symbol_section.hpp>

namespace elf {

class Writer;

namespace writer {

class RelocationSection;

class Relocation {
public:
    using Ptr = std::unique_ptr<Relocation>;

public:
    Elf64_Addr getOffset() const;
    void setOffset(Elf64_Addr offset);

    Elf_Word getType() const;
    void setType(Elf_Word type);

    Elf_Sxword getAddend() const;
    void setAddend(Elf_Sxword addend);

    const Symbol* getSymbol() const;
    void setSymbol(const Symbol* symbol);

private:
    Relocation() = default;

    RelocationAEntry m_relocation{};
    const Symbol* m_symbol = nullptr;

    friend RelocationSection;
};

class RelocationSection final : public Section {
public:
    const SymbolSection* getSymbolTable() const;
    void setSymbolTable(const SymbolSection* symTab);

    const Section* getSectionToPatch() const;
    void setSectionToPatch(const Section* sectionToPatch);

    Relocation* addRelocationEntry();
    const std::vector<Relocation::Ptr>& getRelocations() const;

private:
    RelocationSection();

    void finalize() override;

private:
    const SymbolSection* m_symTab = nullptr;
    const Section* m_sectionToPatch = nullptr;

    std::vector<Relocation::Ptr> m_relocations;

    friend Writer;
};

} // namespace writer
} // namespace elf
