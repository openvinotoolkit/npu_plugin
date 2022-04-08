//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

void vpux::ELF::PadOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto padSize = paddingSize();

    auto padValue = paddingValue().getValueOr(0);

    SmallVector<uint8_t> padding(padSize, padValue);

    binDataSection.appendData(padding.data(), padSize);
}

size_t vpux::ELF::PadOp::getBinarySize() {
    return paddingSize();
}

size_t vpux::ELF::PadOp::getAlignmentRequirements() {
    return ELF::VPUX_NO_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::ELF::PadOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::ELF::PadOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

vpux::ELF::SectionFlagsAttr vpux::ELF::PadOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}
