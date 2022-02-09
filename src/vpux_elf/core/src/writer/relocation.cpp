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

#include <vpux_elf/writer/relocation.hpp>

using namespace elf;
using namespace elf::writer;

//
// RelocationEntry
//

Elf64_Addr Relocation::getOffset() const {
    return m_relocation.r_offset;
}

void Relocation::setOffset(Elf64_Addr offset) {
    m_relocation.r_offset = offset;
}

void Relocation::setType(Elf_Word type) {
    m_relocation.r_info = elf64RInfo(elf64RSym(m_relocation.r_info), type);
}

Elf_Word Relocation::getType() const {
    return elf64RType(m_relocation.r_info);
}

void Relocation::setAddend(Elf_Sxword addend) {
    m_relocation.r_addend = addend;
}

Elf_Sxword Relocation::getAddend() const {
    return m_relocation.r_addend;
}

void Relocation::setSymbol(const Symbol* symbol) {
    m_symbol = symbol;
}

const Symbol* Relocation::getSymbol() const {
    return m_symbol;
}
