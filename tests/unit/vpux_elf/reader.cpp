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

} // namespace

// #-22043
TEST(DISABLED_ELFReaderTests, ELFReaderThrowsOnIncorrectMagic) {
    std::vector<uint8_t> elf = {0x7f, 'E', 'L', 'D'};
    ASSERT_ANY_THROW(Reader(elf.data(), elf.size()));
}

TEST(ELFReaderTests, ReadingTheCorrectELFHeaderDoesntThrow) {
    auto fileHeader = createTemplateFileHeader();

    ASSERT_NO_THROW(Reader(reinterpret_cast<uint8_t*>(&fileHeader), sizeof(fileHeader)));
}

TEST(ELFReaderTests, PointerToELFHeaderIsResolvedCorrectly) {
    auto fileHeader = createTemplateFileHeader();

    const auto reader = Reader(reinterpret_cast<uint8_t*>(&fileHeader), sizeof(fileHeader));
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
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto reader = Reader(buffer.data(), buffer.size());
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
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()),
                  reinterpret_cast<uint8_t*>(sectionHeaders.data()) + sizeof(SectionHeader) * headerTableSize);

    auto reader = Reader(buffer.data(), buffer.size());
    ASSERT_EQ(reinterpret_cast<const uint8_t*>(reader.getSection(indexToCheck).getData<uint8_t>()),
              buffer.data() + sizeof(fileHeader));
}

TEST(ELFReaderTests, PointerToProgramHeaderIsResolvedCorrectly) {
    std::vector<ProgramHeader> programHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_phnum = headerTableSize;
    fileHeader.e_phoff = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(programHeaders.data()),
                  reinterpret_cast<uint8_t*>(programHeaders.data()) + sizeof(ProgramHeader) * headerTableSize);

    auto reader = Reader(buffer.data(), buffer.size());
    ASSERT_EQ(reinterpret_cast<const uint8_t*>(reader.getSegment(indexToCheck).getHeader()),
              buffer.data() + sizeof(fileHeader) + sizeof(ProgramHeader) * indexToCheck);
}

TEST(ELFReaderTests, PointerToProgramDataIsResolvedCorrectly) {
    std::vector<ProgramHeader> programHeaders(headerTableSize);

    auto fileHeader = createTemplateFileHeader();
    fileHeader.e_phnum = headerTableSize;
    fileHeader.e_phoff = sizeof(fileHeader);
    programHeaders[indexToCheck].p_offset = sizeof(fileHeader);

    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&fileHeader),
                  reinterpret_cast<uint8_t*>(&fileHeader) + sizeof(fileHeader));
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(programHeaders.data()),
                  reinterpret_cast<uint8_t*>(programHeaders.data()) + sizeof(ProgramHeader) * headerTableSize);

    auto reader = Reader(buffer.data(), buffer.size());
    ASSERT_EQ(reinterpret_cast<const uint8_t*>(reader.getSegment(indexToCheck).getData()),
              buffer.data() + sizeof(fileHeader));
}
