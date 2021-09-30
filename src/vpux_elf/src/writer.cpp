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

#include <elf/writer.hpp>

using namespace elf;

//
// Writer
//

void Writer::write(const std::string&) {
    // TODO: not implemented
}

void Writer::write(std::ostream&) {
    // TODO: not implemented
}

Elf_Half Writer::getType() const {
    return m_elfHeader.e_type;
}

void Writer::setType(Elf_Half type) {
    m_elfHeader.e_type = type;
}

//
// Writer::Section
//

Elf_Half Writer::Section::getType() const {
    return m_sectionHeader.sh_type;
}

void Writer::Section::setType(Elf_Half type) {
    m_sectionHeader.sh_type = type;
}

//
// Writer::Segment
//

Elf_Half Writer::Segment::getType() const {
    return m_programHeader.p_type;
}

void Writer::Segment::setType(Elf_Half type) {
    m_programHeader.p_type = type;
}
