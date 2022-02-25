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

#include <vpux_elf/writer/section.hpp>

using namespace elf;
using namespace elf::writer;

Section::Section() {
    m_header.sh_name = 0;
    m_header.sh_type = SHT_NULL;
    m_header.sh_flags = 0;
    m_header.sh_addr = 0;
    m_header.sh_offset = 0;
    m_header.sh_size = 0;
    m_header.sh_link = SHN_UNDEF;
    m_header.sh_info = 0;
    m_header.sh_addralign = 4;
    m_header.sh_entsize = 0;
}

std::string Section::getName() const {
    return m_name;
}

void Section::setName(const std::string& name) {
    m_name = name;
}

Elf_Xword Section::getAddrAlign() const {
    return m_header.sh_addralign;
}

void Section::setAddrAlign(Elf_Xword addrAlign) {
    m_header.sh_addralign = addrAlign;
}

Elf64_Addr Section::getAddr() const {
    return m_header.sh_addr;
}

void Section::setAddr(Elf64_Addr addr) {
    m_header.sh_addr = addr;
}

Elf_Xword Section::getFlags() const {
    return m_header.sh_flags;
}

void Section::setFlags(Elf_Xword flags) {
    m_header.sh_flags = flags;
}

void Section::maskFlags(Elf_Xword flags) {
    m_header.sh_flags |= flags;
}

size_t Section::getFileAlignRequirement() const {
    return m_fileAlignRequirement;
}

void Section::finalize() {}

void Section::setIndex(size_t index) {
    m_index = index;
}

void Section::setNameOffset(size_t offset) {
    m_header.sh_name = offset;
}

size_t Section::getIndex() const {
    return m_index;
}

size_t Section::getDataSize() const {
    return m_data.size();
}
