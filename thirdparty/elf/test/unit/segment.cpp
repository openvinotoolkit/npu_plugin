// {% copyright %}

#include <elf/elf.hpp>

#include <gtest/gtest.h>

TEST(ELF_Segment, AddingSegmentDoesntThrow) {
    elf::ELF elf;
    ASSERT_NO_THROW(elf.addSection());
}

TEST(ELF_Segment, SegmentTypeIsChangingAfterWriting) {
    elf::ELF elf;
    const auto segment = elf.addSegment();
    segment->setType(elf::PT_LOAD);
    ASSERT_EQ(segment->getType(), elf::PT_LOAD);
}
