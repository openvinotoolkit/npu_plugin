// {% copyright %}

#include <elf/elf.hpp>

#include <gtest/gtest.h>

TEST(ELF_Section, AddingSectionDoesntThrow) {
    elf::ELF elf;
    ASSERT_NO_THROW(elf.addSection());
}

TEST(ELF_Section, SectionTypeIsChangingAfterWriting) {
    elf::ELF elf;
    const auto section = elf.addSection();
    section->setType(elf::SHT_PROGBITS);
    ASSERT_EQ(section->getType(), elf::SHT_PROGBITS);
}
