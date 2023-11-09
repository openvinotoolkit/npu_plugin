//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>

#include <gtest/gtest.h>

using namespace elf;

namespace {

ELFHeader createTemplateFileHeader() {
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
    fileHeader.e_shoff = 0;
    fileHeader.e_phoff = 0;
    fileHeader.e_shstrndx = 0;
    fileHeader.e_shnum = 0;
    fileHeader.e_phnum = 0;

    fileHeader.e_ehsize = sizeof(ELFHeader);
    fileHeader.e_phentsize = sizeof(ProgramHeader);
    fileHeader.e_shentsize = sizeof(SectionHeader);

    return fileHeader;
}

}  // namespace

TEST(ELFReaderTests, ELFReaderThrowsOnIncorrectMagic) {
    std::vector<uint8_t> elf = {0x7f, 'E', 'L', 'D'};
    auto accessor = elf::ElfDDRAccessManager(elf.data(), elf.size());

    ASSERT_ANY_THROW(auto reader = Reader<ELF_Bitness::Elf64>(&accessor));
}

TEST(ELFReaderTests, ReadingTheCorrectELFHeaderDoesntThrow) {
    auto fileHeader = createTemplateFileHeader();
    auto accessor = elf::ElfDDRAccessManager(reinterpret_cast<uint8_t*>(&fileHeader), sizeof(fileHeader));

    ASSERT_NO_THROW(auto reader = Reader<ELF_Bitness::Elf64>(&accessor));
}

TEST(ELFReaderTests, PointerToELFHeaderIsResolvedCorrectly) {
    auto fileHeader = createTemplateFileHeader();

    auto accessor = elf::ElfDDRAccessManager(reinterpret_cast<uint8_t*>(&fileHeader), sizeof(fileHeader));
    const auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    const auto parsedFileHeader = reader.getHeader();

    ASSERT_EQ(parsedFileHeader, &fileHeader);
}

constexpr size_t headerTableSize = 3;
constexpr size_t indexToCheck = 1;

TEST(ELFReaderTests, PointerToSectionHeaderIsResolvedCorrectly) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    fileHeader.e_shoff = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto accessor = elf::ElfDDRAccessManager(buffer.data(), buffer.size());
    auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    ASSERT_EQ(reinterpret_cast<const uint8_t*>(reader.getSection(indexToCheck).getHeader()),
              buffer.data() + sizeof(fileHeader) + sizeof(SectionHeader) * indexToCheck);
}

TEST(ELFReaderTests, PointerToSectionDataIsResolvedCorrectly) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    fileHeader.e_shoff = sizeof(fileHeader);
    sectionHeaders[indexToCheck].sh_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto accessor = elf::ElfDDRAccessManager(buffer.data(), buffer.size());
    auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    ASSERT_EQ(reader.getSection(indexToCheck).getData<uint8_t>(), buffer.data() + sizeof(fileHeader));
}

TEST(ELFReaderTests, PtrToSectionDataIsResolvedCorrectlyWithGetSectionNoData) {
    std::vector<SectionHeader> sectionHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_shnum = headerTableSize;
    fileHeader.e_shoff = sizeof(fileHeader);
    sectionHeaders[indexToCheck].sh_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto accessor = elf::ElfDDRAccessManager(buffer.data(), buffer.size());
    auto reader = Reader<ELF_Bitness::Elf64>(&accessor);
    ASSERT_EQ(reader.getSectionNoData(indexToCheck).getData<uint8_t>(), buffer.data() + sizeof(fileHeader));
}
