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

#include <elf/reader.hpp>

using namespace elf;

//
// Reader
//

Reader::Reader(const std::string&) {
    // TODO: implement
}

Reader::Reader(std::istream&) {
    // TODO: implement
}

Reader::Reader(std::vector<char>) {
    // TODO: implement
}

Elf_Half Reader::getType() const {
    return m_elfHeader->e_type;
}

//
// Reader::Section
//

Reader::Section::Section(const char* sectionHeader, const char* data) {
    m_sectionHeader = reinterpret_cast<const Elf64_Shdr*>(sectionHeader);
    m_data = data;
}

Elf_Half Reader::Section::getType() const {
    return m_sectionHeader->sh_type;
}

//
// Reader::Segment
//

Reader::Segment::Segment(const char* programHeader, const char* data) {
    m_programHeader = reinterpret_cast<const Elf64_Phdr*>(programHeader);
    m_data = data;
}

Elf_Half Reader::Segment::getType() const {
    return m_programHeader->p_type;
}
