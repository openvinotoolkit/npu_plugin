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
