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

#include <unordered_set>
#include <algorithm>

using namespace elf;
using namespace elf::writer;

namespace {

uint64_t alignOffset(uint64_t offset, uint64_t alignReq) {
    return ((offset + (alignReq - 1)) / alignReq) * alignReq;
}

} // namespace

//
// Writer
//

Writer::Writer() {
    // Creating NULL section
    m_sections.push_back(std::unique_ptr<Section>(new Section));

    m_sectionHeaderNames = addStringSection(".strtab");

    m_symbolNames = addStringSection(".symstrtab");
};

std::vector<uint8_t> Writer::generateELF() {
    auto elfHeader = generateELFHeader();

    std::vector<elf::SectionHeader> sectionHeaders;
    sectionHeaders.reserve(elfHeader.e_shnum);
    std::vector<elf::ProgramHeader> programHeaders;
    programHeaders.reserve(elfHeader.e_phnum);

    std::vector<Section*> sectionsFromSegments;
    for (const auto& segment : m_segments) {
        for (const auto& section : segment->m_sections) {
            sectionsFromSegments.push_back(section);
        }
    }

    elfHeader.e_shstrndx = m_sectionHeaderNames->getIndex();

    for (auto& section : m_sections) {
        section->finalize();
        section->setNameOffset(m_sectionHeaderNames->addString(section->getName()));
    }

    auto curOffset = elfHeader.e_ehsize;
    if (elfHeader.e_shnum) {
        curOffset = elfHeader.e_shoff = alignOffset(curOffset, elfHeader.e_shentsize);
    }
    if (elfHeader.e_phnum) {
        curOffset = elfHeader.e_phoff = alignOffset(curOffset + elfHeader.e_shnum * elfHeader.e_shentsize, elfHeader.e_phentsize);
    } else {
        curOffset += elfHeader.e_shnum * elfHeader.e_shentsize;
    }
    const auto dataOffset = curOffset + elfHeader.e_phnum * elfHeader.e_phentsize;

    std::vector<uint8_t> data;

    const auto alignData = [&data, &dataOffset](const Section* section) {
        const auto curFileOffset = dataOffset + data.size();
        const auto alignedFileOffset = alignOffset(curFileOffset, section->getFileAlignRequirement());
        if (curFileOffset != alignedFileOffset) {
            data.resize(data.size() + (alignedFileOffset - curFileOffset), 0);
        }
    };

    const auto serializeSection = [&data, &dataOffset, &sectionHeaders, &alignData](Section* section) {
        const auto sectionData = section->m_data;
        auto sectionHeader = section->m_header;

        alignData(section);

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

    std::vector<uint8_t> elfBlob;
    elfBlob.reserve(dataOffset + data.size());

    elfBlob.insert(elfBlob.end(), reinterpret_cast<uint8_t*>(&elfHeader), reinterpret_cast<uint8_t*>(&elfHeader) + elfHeader.e_ehsize);
    if (elfHeader.e_shoff) {
        elfBlob.resize(elfHeader.e_shoff, 0);
        elfBlob.insert(elfBlob.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                       reinterpret_cast<uint8_t*>(sectionHeaders.data()) + elfHeader.e_shnum * elfHeader.e_shentsize);
    }
    if (elfHeader.e_phoff) {
        elfBlob.resize(elfHeader.e_phoff, 0);
        elfBlob.insert(elfBlob.end(), reinterpret_cast<uint8_t*>(programHeaders.data()),
                       reinterpret_cast<uint8_t*>(programHeaders.data()) + elfHeader.e_phnum * elfHeader.e_phentsize);
    }
    elfBlob.insert(elfBlob.end(), data.data(), data.data() + data.size());

    return elfBlob;
}

Segment* Writer::addSegment() {
    m_segments.push_back(std::unique_ptr<Segment>(new Segment));
    return m_segments.back().get();
}

Section* Writer::addSection(const std::string& name) {
    m_sections.push_back(std::unique_ptr<Section>(new Section(name)));
    m_sections.back()->setIndex(m_sections.size() - 1);
    return m_sections.back().get();
}

RelocationSection* Writer::addRelocationSection(const std::string& name) {
    m_sections.push_back(std::unique_ptr<RelocationSection>(new RelocationSection(name)));
    m_sections.back()->setIndex(m_sections.size() - 1);
    return dynamic_cast<RelocationSection*>(m_sections.back().get());
}

SymbolSection* Writer::addSymbolSection(const std::string& name) {
    m_sections.push_back(std::unique_ptr<SymbolSection>(new SymbolSection(name, m_symbolNames)));
    m_sections.back()->setIndex(m_sections.size() - 1);
    return dynamic_cast<SymbolSection*>(m_sections.back().get());
}

EmptySection* Writer::addEmptySection(const std::string& name) {
    m_sections.push_back(std::unique_ptr<EmptySection>(new EmptySection(name)));
    m_sections.back()->setIndex(m_sections.size() - 1);
    return dynamic_cast<EmptySection*>(m_sections.back().get());
}

StringSection* Writer::addStringSection(const std::string& name) {
    m_sections.push_back(std::unique_ptr<StringSection>(new StringSection(name)));
    m_sections.back()->setIndex(m_sections.size() - 1);
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

    fileHeader.e_shoff = fileHeader.e_phoff = 0;

    fileHeader.e_ehsize = sizeof(ELFHeader);
    fileHeader.e_phentsize = sizeof(ProgramHeader);
    fileHeader.e_shentsize = sizeof(SectionHeader);

    return fileHeader;
}
