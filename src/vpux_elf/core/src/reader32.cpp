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

#include <vpux_elf/reader32.hpp>

using namespace elf;

//
// Reader
//


Reader32::Reader32(){}

void Reader32::loadElf(const char* blob, size_t) {
    m_blob = blob;
    m_elfHeader = reinterpret_cast<decltype(m_elfHeader)>(blob);
    m_sectionHeadersStart = reinterpret_cast<const SectionHeader32*>(m_blob + m_elfHeader->e_shoff);
    m_programHeadersStart = reinterpret_cast<const ProgramHeader32*>(m_blob + m_elfHeader->e_phoff);
    m_sectionHeadersNames = m_blob + (m_sectionHeadersStart + m_elfHeader->e_shstrndx)->sh_offset;
}

const char* Reader32::getBlob() const {
    return m_blob;
}

const ELF32Header* Reader32::getHeader() const {
    return m_elfHeader;
}

size_t Reader32::getSectionsNum() const {
    return m_elfHeader->e_shnum;
}

size_t Reader32::getSegmentsNum() const {
    return m_elfHeader->e_phnum;
}

Reader32::Section Reader32::getSection(size_t index) {
    const auto sectionHeader = m_sectionHeadersStart + index;
    auto data = m_blob + sectionHeader->sh_offset;
    const auto name = m_sectionHeadersNames + sectionHeader->sh_name;

    return {sectionHeader, data, name};
}

Reader32::Segment Reader32::getSegment(size_t index) {
    const auto programHeader = m_programHeadersStart + index;
    auto data = m_blob + programHeader->p_offset;

    return {programHeader, data};
}

//
// Reader::Section
//

Reader32::Section::Section(const SectionHeader32* sectionHeader, const char* data, const char* name) :
      m_sectionHeader(sectionHeader), m_data(data), m_name(name) {}

const SectionHeader32* Reader32::Section::getHeader() const {
    return m_sectionHeader;
}

size_t Reader32::Section::getEntriesNum() const {
    return static_cast<size_t>(m_sectionHeader->sh_size / m_sectionHeader->sh_entsize);
}

const char* Reader32::Section::getName() const {
    return m_name;
}

//
// Reader::Segment
//

Reader32::Segment::Segment(const ProgramHeader32* programHeader, const char* data) : m_programHeader(programHeader), m_data(data) {}

const ProgramHeader32* Reader32::Segment::getHeader() const {
    return m_programHeader;
}

const char* Reader32::Segment::getData() const {
    return m_data;
}
