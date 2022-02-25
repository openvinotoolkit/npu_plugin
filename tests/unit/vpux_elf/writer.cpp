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

#include <vpux_elf/writer.hpp>
#include <vpux_elf/reader.hpp>

#include <gtest/gtest.h>

namespace {

std::string getSectionName(elf::Reader& reader, const elf::Reader::Section& section) {
    const auto elfHeader = reader.getHeader();
    const auto testNames = reader.getSection(elfHeader->e_shstrndx);
    const auto strings = testNames.getData<char>();
    return std::string(strings + section.getHeader()->sh_name);
}

std::string getSymbolName(elf::Reader& reader, const elf::Reader::Section& symbolSection, const elf::SymbolEntry& symbol) {
    const auto symStrTab = reader.getSection(symbolSection.getHeader()->sh_link);
    const auto strings = symStrTab.getData<char>();
    return std::string(strings + symbol.st_name);
}

std::vector<elf::Reader::Section> getSectionsByType(elf::Reader& reader, elf::Elf_Word type) {
    std::vector<elf::Reader::Section> res;

    for (size_t i = 0; i < reader.getSectionsNum(); ++i) {
        const auto& section = reader.getSection(i);
        if (section.getHeader()->sh_type == type) {
            res.push_back(section);
        }
    }

    return res;
}

} // namespace

TEST(ELFWriter, ELFWriterConstructorDoesntThrow) {
    ASSERT_NO_THROW(elf::Writer());
}

TEST(ELFWriter, ELFHeaderForEmptyELFIsCorrect) {
    elf::Writer writer;
    std::vector<uint8_t> blob;
    ASSERT_NO_THROW(blob = writer.generateELF());

    elf::Reader reader(blob.data(), blob.size());
    const auto elfHeader = reader.getHeader();
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG0], elf::ELFMAG0);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG1], elf::ELFMAG1);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG2], elf::ELFMAG2);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_MAG3], elf::ELFMAG3);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_CLASS], elf::ELFCLASS64);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_DATA], elf::ELFDATA2LSB);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_VERSION], elf::EV_NONE);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_OSABI], 0);
    ASSERT_EQ(elfHeader->e_ident[elf::EI_ABIVERSION], 0);
    ASSERT_EQ(elfHeader->e_type, elf::ET_REL);
    ASSERT_EQ(elfHeader->e_machine, elf::EM_NONE);
    ASSERT_EQ(elfHeader->e_version, elf::EV_NONE);
    ASSERT_EQ(elfHeader->e_entry, 0);
    ASSERT_EQ(elfHeader->e_flags, 0);
    ASSERT_EQ(elfHeader->e_ehsize, sizeof(elf::ELFHeader));
    ASSERT_EQ(elfHeader->e_phentsize, sizeof(elf::ProgramHeader));
    ASSERT_EQ(elfHeader->e_shentsize, sizeof(elf::SectionHeader));
}

TEST(ELFWriter, BinaryDataSection) {
    constexpr int testAlignment = 64;
    const auto testName = std::string(".test");

    struct TestObject {
        int a;
        float b;
    };
    const auto val1 = TestObject{0, 42};
    const auto val2 = TestObject{42, 0};

    elf::Writer writer;
    auto refSection = writer.addBinaryDataSection<TestObject>();
    refSection->setName(testName);
    refSection->setAddrAlign(testAlignment);
    refSection->appendData(val1);
    refSection->appendData(val2);

    std::vector<uint8_t> blob;
    ASSERT_NO_THROW(blob = writer.generateELF());

    elf::Reader reader(blob.data(), blob.size());
    const auto binarySections = getSectionsByType(reader, elf::SHT_PROGBITS);
    ASSERT_EQ(binarySections.size(), 1);
    ASSERT_EQ(reader.getSegmentsNum(), 0);

    const auto& binarySection = binarySections.front();
    ASSERT_EQ(getSectionName(reader, binarySection), testName);
    ASSERT_EQ(binarySection.getHeader()->sh_addralign, testAlignment);
    ASSERT_EQ(binarySection.getHeader()->sh_entsize, sizeof(TestObject));
    ASSERT_EQ(binarySection.getHeader()->sh_size, sizeof(TestObject) * 2);

    const auto binarySectionData = binarySection.getData<TestObject>();
    ASSERT_EQ(std::tie(binarySectionData[0].a, binarySectionData[0].b), std::tie(val1.a, val1.b));
    ASSERT_EQ(std::tie(binarySectionData[1].a, binarySectionData[1].b), std::tie(val2.a, val2.b));
}

TEST(ELFWriter, EmptySection) {
    const auto testName = std::string(".test");
    constexpr int emptySectionSize = 42;
    constexpr auto emptySectionFlags = elf::SHF_ALLOC;

    elf::Writer writer;
    auto refSection = writer.addEmptySection();
    refSection->setName(testName);
    refSection->setFlags(emptySectionFlags);
    refSection->setSize(emptySectionSize);

    std::vector<uint8_t> blob;
    ASSERT_NO_THROW(blob = writer.generateELF());

    elf::Reader reader(blob.data(), blob.size());
    const auto emptySections = getSectionsByType(reader, elf::SHT_NOBITS);
    ASSERT_EQ(emptySections.size(), 1);
    ASSERT_EQ(reader.getSegmentsNum(), 0);

    const auto& emptySection = emptySections.front();
    ASSERT_EQ(getSectionName(reader, emptySection), testName);
    ASSERT_EQ(emptySection.getHeader()->sh_size, emptySectionSize);
    ASSERT_EQ(emptySection.getHeader()->sh_flags, emptySectionFlags);
}

TEST(ELFWriter, SymbolSection) {
    const auto testName = std::string(".test");
    constexpr int symbolSize = 42;
    constexpr int symbolValue = 2;
    constexpr auto symbolType = elf::STT_SECTION;

    elf::Writer writer;
    auto emptySection = writer.addEmptySection();

    auto refSection = writer.addSymbolSection();
    refSection->setName(testName);

    auto refSymbol = refSection->addSymbolEntry();
    refSymbol->setName(testName);
    refSymbol->setValue(symbolValue);
    refSymbol->setSize(symbolSize);
    refSymbol->setType(symbolType);
    refSymbol->setRelatedSection(emptySection);

    std::vector<uint8_t> blob;
    ASSERT_NO_THROW(blob = writer.generateELF());

    elf::Reader reader(blob.data(), blob.size());
    const auto symbolSections = getSectionsByType(reader, elf::SHT_SYMTAB);
    ASSERT_EQ(symbolSections.size(), 1);
    ASSERT_EQ(reader.getSegmentsNum(), 0);

    const auto& symbolSection = symbolSections.front();
    ASSERT_EQ(getSectionName(reader, symbolSection), testName);
    ASSERT_EQ(symbolSection.getHeader()->sh_entsize, sizeof(elf::SymbolEntry));
    ASSERT_EQ(symbolSection.getHeader()->sh_size, sizeof(elf::SymbolEntry) * 2);

    const auto symbol = symbolSection.getData<elf::SymbolEntry>()[1];

    ASSERT_EQ(getSymbolName(reader, symbolSection, symbol), testName);
    ASSERT_EQ(symbol.st_value, symbolValue);
    ASSERT_EQ(symbol.st_size, symbolSize);
    ASSERT_EQ(elf::elf64STType(symbol.st_info), symbolType);

    const auto relatedSection = reader.getSection(symbol.st_shndx);
    ASSERT_EQ(relatedSection.getHeader()->sh_type, elf::SHT_NOBITS);
}

TEST(ELFWriter, RelocationSection) {
    const auto testBinaryDataName = std::string(".test.BinaryData");
    const auto testRelocationName = std::string(".test.Relocation");
    const auto testSymbolSection = std::string(".test.Symbols");
    const auto testSymbolName = std::string(".test.Symbol");
    constexpr auto testSymbolType = elf::STT_SECTION;
    constexpr auto testSymbolSize = 42;
    struct TestObject {
        uint32_t a = 0;
        uint64_t b = 0;
    };

    elf::Writer writer;
    auto refBinaryDataSection = writer.addBinaryDataSection<TestObject>();
    refBinaryDataSection->setName(testBinaryDataName);
    refBinaryDataSection->appendData(TestObject{});

    auto refSymbolSection = writer.addSymbolSection();
    refSymbolSection->setName(testSymbolSection);
    auto refSymbol = refSymbolSection->addSymbolEntry();
    refSymbol->setName(testSymbolName);
    refSymbol->setType(testSymbolType);
    refSymbol->setSize(testSymbolSize);

    auto refRelocationSection = writer.addRelocationSection();
    refRelocationSection->setName(testRelocationName);
    refRelocationSection->setSectionToPatch(refBinaryDataSection);
    refRelocationSection->setSymbolTable(refSymbolSection);
    auto refRelocation = refRelocationSection->addRelocationEntry();
    refRelocation->setSymbol(refSymbol);
    refRelocation->setOffset(sizeof(TestObject::a));
    refRelocation->setAddend(0);

    std::vector<uint8_t> blob;
    ASSERT_NO_THROW(blob = writer.generateELF());

    elf::Reader reader(blob.data(), blob.size());
    const auto relocationSections = getSectionsByType(reader, elf::SHT_RELA);
    ASSERT_EQ(relocationSections.size(), 1);
    ASSERT_EQ(reader.getSegmentsNum(), 0);

    const auto& relocationSection = relocationSections.front();
    ASSERT_EQ(getSectionName(reader, relocationSection), testRelocationName);
    ASSERT_EQ(relocationSection.getHeader()->sh_entsize, sizeof(elf::RelocationAEntry));
    ASSERT_EQ(relocationSection.getHeader()->sh_size, sizeof(elf::RelocationAEntry));

    const auto relocationToPatch = reader.getSection(relocationSection.getHeader()->sh_info);
    ASSERT_EQ(getSectionName(reader, relocationToPatch), testBinaryDataName);
    ASSERT_EQ(relocationToPatch.getHeader()->sh_type, elf::SHT_PROGBITS);

    const auto symbolTable = reader.getSection(relocationSection.getHeader()->sh_link);
    ASSERT_EQ(getSectionName(reader, symbolTable), testSymbolSection);
    ASSERT_EQ(symbolTable.getHeader()->sh_type, elf::SHT_SYMTAB);

    const auto relocation = relocationSection.getData<elf::RelocationAEntry>()[0];
    ASSERT_EQ(relocation.r_addend, 0);
    ASSERT_EQ(relocation.r_offset, sizeof(TestObject::a));

    const auto symbol = symbolTable.getData<elf::SymbolEntry>()[elf::elf64RSym(relocation.r_info)];
    ASSERT_EQ(getSymbolName(reader, symbolTable, symbol), testSymbolName);
    ASSERT_EQ(elf::elf64STType(symbol.st_info), testSymbolType);
    ASSERT_EQ(symbol.st_size, testSymbolSize);
}

TEST(ELFWriter, Segment) {
    constexpr auto testSegmentType = elf::PT_LOAD;
    const auto testSectionData = std::vector<uint8_t>{0, 1, 2, 3};
    const auto testSegmentData = std::vector<uint8_t>{4, 5, 6, 7};

    elf::Writer writer;
    auto refSegment = writer.addSegment();
    refSegment->setType(testSegmentType);
    refSegment->appendData(testSegmentData.data(), testSegmentData.size());

    auto section = writer.addBinaryDataSection<uint8_t>();
    section->appendData(testSectionData.data(), testSectionData.size());
    refSegment->addSection(section);

    std::vector<uint8_t> blob;
    ASSERT_NO_THROW(blob = writer.generateELF());
    elf::Reader reader(blob.data(), blob.size());
    ASSERT_EQ(reader.getSegmentsNum(), 1);

    const auto segment = reader.getSegment(0);
    ASSERT_EQ(segment.getHeader()->p_type, testSegmentType);
    ASSERT_EQ(segment.getHeader()->p_filesz, testSectionData.size() + testSegmentData.size());
    ASSERT_EQ(segment.getHeader()->p_filesz, segment.getHeader()->p_memsz);

    const auto segmentData = segment.getData();
    for (size_t i = 0; i < testSectionData.size(); ++i) {
        ASSERT_EQ(testSectionData[i], segmentData[i]);
    }
    for (size_t i = 0; i < testSegmentData.size(); ++i) {
        ASSERT_EQ(testSegmentData[i], segmentData[i] + testSectionData.size());
    }
}
