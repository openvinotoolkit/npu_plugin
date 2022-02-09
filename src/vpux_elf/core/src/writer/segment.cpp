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

#include <vpux_elf/writer/segment.hpp>

#include <vpux/utils/core/error.hpp>

using namespace elf;
using namespace elf::writer;

Segment::Segment() {
    m_header.p_type = PT_NULL;
    m_header.p_flags = 0;
    m_header.p_offset = 0;
    m_header.p_vaddr = 0;
    m_header.p_paddr = 0;
    m_header.p_filesz = 0;
    m_header.p_memsz = 0;
    m_header.p_align = 0;
}

void Segment::addSection(Section* section) {
    VPUX_THROW_UNLESS(section->getFileAlignRequirement() == 1, "Adding section with file offset requirement {0} is not supported",
                      section->getFileAlignRequirement());
    m_sections.push_back(section);
}

void Segment::setType(Elf_Word type) {
    m_header.p_type = type;
}

void Segment::setAlign(Elf_Xword align) {
    m_header.p_align = align;
}
