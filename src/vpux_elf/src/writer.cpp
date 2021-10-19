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

#include <fstream>
#include <unordered_set>
#include <algorithm>

#include <iostream>

using namespace elf;
using namespace elf::writer;

//
// Writer
//

Writer::Writer() {
    // Creating NULL section
    m_sections.push_back(Section::Ptr(new Section));

    m_sectionHeaderNames = addStringSection();
    m_sectionHeaderNames->setName(".strtab");

    m_symbolNames = addStringSection();
    m_symbolNames->setName(".symstrtab");
};

void Writer::write(const std::string& fileName) {
    std::ofstream stream(fileName, std::ios::out | std::ios::binary);
    write(stream);
    stream.close();
}

void Writer::write(std::ostream& stream) {
    std::vector<char> blob;
    write(blob);
    stream.write(blob.data(), blob.size());
}

void Writer::write(std::vector<char>& blob) {
    blob.clear();

    auto elfHeader = generateELFHeader();

    std::vector<elf::SectionHeader> sectionHeaders;
    sectionHeaders.reserve(elfHeader.e_shnum);
    std::vector<elf::ProgramHeader> programHeaders;
    programHeaders.reserve(elfHeader.e_phnum);

    const auto elfHeaderSize = elfHeader.e_ehsize;
    const auto sectionHeadersSize = elfHeader.e_shnum * elfHeader.e_shentsize;
    const auto programHeadersSize = elfHeader.e_phnum * elfHeader.e_phentsize;
    const auto dataOffset = elfHeaderSize + sectionHeadersSize + programHeadersSize;

    std::vector<char> data;

    std::vector<Section*> sectionsFromSegments;
    for (const auto& segment : m_segments) {
        for (const auto& section : segment->m_sections) {
            sectionsFromSegments.push_back(section);
        }
    }

    int curIndex = 0;
    for (auto& section : m_sections) {
        if (std::find(sectionsFromSegments.begin(), sectionsFromSegments.end(), section.get()) == sectionsFromSegments.end()) {
            section->setIndex(curIndex++);
        }
    }
    for (auto& section : sectionsFromSegments) {
        section->setIndex(curIndex++);
    }

    elfHeader.e_shstrndx = m_sectionHeaderNames->getIndex();

    for (auto& section : m_sections) {
        section->finalize();
        section->setNameOffset(m_sectionHeaderNames->addString(section->getName()));
    }

    const auto serializeSection = [&data, &dataOffset, &sectionHeaders](Section* section) {
        const auto sectionData = section->m_data;
        auto sectionHeader = section->m_header;

        if (!sectionData.empty()) {
            sectionHeader.sh_offset = dataOffset + data.size();
            sectionHeader.sh_size = sectionData.size();
        }
        sectionHeaders.push_back(sectionHeader);
        data.insert(data.end(), sectionData.data(), sectionData.data() + sectionData.size());
    };

    for (auto& section : m_sections) {
        if (std::find(sectionsFromSegments.begin(), sectionsFromSegments.end(), section.get()) != sectionsFromSegments.end()) {
            continue;
        }
        serializeSection(section.get());
    }

    for (auto& segment : m_segments) {
        if (segment->m_data.empty() && segment->m_sections.empty()) {
            continue;
        }

        auto programHeader = segment->m_header;
        programHeader.p_offset = dataOffset + data.size();

        for (auto& section : segment->m_sections) {
            programHeader.p_filesz += section->m_data.size();
            serializeSection(section);
        }

        if (!segment->m_data.empty()) {
            programHeader.p_filesz += segment->m_data.size();
            data.insert(data.end(), segment->m_data.data(), segment->m_data.data() + segment->m_data.size());
        }

        programHeader.p_memsz = programHeader.p_filesz;

        programHeaders.push_back(programHeader);
    }

    blob.reserve(elfHeaderSize + sectionHeadersSize + programHeadersSize + data.size());

    blob.insert(blob.end(), reinterpret_cast<char*>(&elfHeader), reinterpret_cast<char*>(&elfHeader) + elfHeaderSize);
    blob.insert(blob.end(), reinterpret_cast<char*>(sectionHeaders.data()), reinterpret_cast<char*>(sectionHeaders.data()) + sectionHeadersSize);
    blob.insert(blob.end(), reinterpret_cast<char*>(programHeaders.data()), reinterpret_cast<char*>(programHeaders.data()) + programHeadersSize);
    blob.insert(blob.end(), data.data(), data.data() + data.size());
}

Segment* Writer::addSegment() {
    m_segments.push_back(std::unique_ptr<Segment>(new Segment));
    return m_segments.back().get();
}

Section* Writer::addSection() {
    m_sections.push_back(std::unique_ptr<Section>(new Section));
    return m_sections.back().get();
}

RelocationSection* Writer::addRelocationSection() {
    m_sections.push_back(std::unique_ptr<RelocationSection>(new RelocationSection));
    return dynamic_cast<RelocationSection*>(m_sections.back().get());
}

SymbolSection* Writer::addSymbolSection() {
    m_sections.push_back(std::unique_ptr<SymbolSection>(new SymbolSection(m_symbolNames)));
    return dynamic_cast<SymbolSection*>(m_sections.back().get());
}

EmptySection* Writer::addEmptySection() {
    m_sections.push_back(std::unique_ptr<EmptySection>(new EmptySection));
    return dynamic_cast<EmptySection*>(m_sections.back().get());
}

StringSection* Writer::addStringSection() {
    m_sections.push_back(std::unique_ptr<StringSection>(new StringSection));
    return dynamic_cast<StringSection*>(m_sections.back().get());
}

elf::ELFHeader Writer::generateELFHeader() const {
    ELFHeader fileHeader;

    fileHeader.e_ident[EI_MAG0] = ELFMAG0;
    fileHeader.e_ident[EI_MAG1] = ELFMAG1;
    fileHeader.e_ident[EI_MAG2] = ELFMAG2;
    fileHeader.e_ident[EI_MAG3] = ELFMAG3;
    fileHeader.e_ident[EI_CLASS] = ELFCLASS64;
    fileHeader.e_ident[EI_DATA] = ELFDATA2LSB;
    fileHeader.e_ident[EI_VERSION] = EV_NONE;
    fileHeader.e_ident[EI_OSABI] = 0;
    fileHeader.e_ident[EI_ABIVERSION] = 0;

    fileHeader.e_type = ET_REL;
    fileHeader.e_machine = EM_NONE;
    fileHeader.e_version = EV_NONE;

    fileHeader.e_entry = 0;
    fileHeader.e_flags = 0;

    fileHeader.e_shnum = m_sections.size();
    fileHeader.e_phnum = m_segments.size();

    fileHeader.e_ehsize = sizeof(ELFHeader);
    fileHeader.e_phentsize = sizeof(ProgramHeader);
    fileHeader.e_shentsize = sizeof(SectionHeader);

    fileHeader.e_shoff = fileHeader.e_ehsize;
    fileHeader.e_phoff = fileHeader.e_shoff + fileHeader.e_shnum * fileHeader.e_shentsize;

    return fileHeader;
}
