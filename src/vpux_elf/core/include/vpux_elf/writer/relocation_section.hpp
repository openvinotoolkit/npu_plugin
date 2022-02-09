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

#include <vpux_elf/writer/section.hpp>

#include <vpux_elf/writer/relocation.hpp>

namespace elf {

class Writer;

namespace writer {

class RelocationSection final : public Section {
public:
    const SymbolSection* getSymbolTable() const;
    void setSymbolTable(const SymbolSection* symTab);

    const Section* getSectionToPatch() const;
    void setSectionToPatch(const Section* sectionToPatch);

    Relocation* addRelocationEntry();
    const std::vector<std::unique_ptr<Relocation>>& getRelocations() const;

private:
    explicit RelocationSection(const std::string& name);

    void finalize() override;

private:
    const SymbolSection* m_symTab = nullptr;
    const Section* m_sectionToPatch = nullptr;

    std::vector<std::unique_ptr<Relocation>> m_relocations;

    friend Writer;
};

} // namespace writer
} // namespace elf
