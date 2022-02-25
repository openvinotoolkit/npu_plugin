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

#include <vpux_elf/writer/empty_section.hpp>

using namespace elf;
using namespace elf::writer;

EmptySection::EmptySection() {
    m_header.sh_type = elf::SHT_NOBITS;
}

Elf_Xword EmptySection::getSize() const {
    return m_header.sh_size;
}

void EmptySection::setSize(Elf_Xword size) {
    m_header.sh_size = size;
}

Elf_Word EmptySection::getType() const {
    return m_header.sh_type;
}

void EmptySection::setType(Elf_Word type) {
    m_header.sh_type = type;
}
