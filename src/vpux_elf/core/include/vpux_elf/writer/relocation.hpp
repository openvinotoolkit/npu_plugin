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

#include <vpux_elf/types/relocation_entry.hpp>
#include <vpux_elf/writer/symbol_section.hpp>

namespace elf {
namespace writer {

class RelocationSection;

class Relocation {
public:
    Elf64_Addr getOffset() const;
    void setOffset(Elf64_Addr offset);

    Elf_Word getType() const;
    void setType(Elf_Word type);

    Elf_Sxword getAddend() const;
    void setAddend(Elf_Sxword addend);

    const Symbol* getSymbol() const;
    void setSymbol(const Symbol* symbol);

    Elf_Word getSpecialSymbol() const;
    void setSpecialSymbol(Elf_Word specialSymbol);

private:
    using Ptr = std::unique_ptr<Relocation>;

private:
    Relocation() = default;

    RelocationAEntry m_relocation{};
    const Symbol* m_symbol = nullptr;

    friend RelocationSection;
};

} // namespace writer
} // namespace elf
