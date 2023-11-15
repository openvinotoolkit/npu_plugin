//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include <flatbuffers/flatbuffers.h>

#include "vpux/compiler/profiling/generated/schema/profiling_generated.h"

using namespace vpux;

//
//  ProfilingMetadataOp
//

void vpux::VPUMI37XX::ProfilingMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                     const flatbuffers::DetachedBuffer& metadata) {
    VPUX_THROW_UNLESS(metadata.data() != nullptr, "ProfilingMetadata's data is NULL");
    binDataSection.appendData(metadata.data(), metadata.size());
}

void vpux::VPUMI37XX::ProfilingMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
    VPUX_THROW("ERROR");
}

size_t vpux::VPUMI37XX::ProfilingMetadataOp::getBinarySize() {
    return sizeof(ProfilingFB::ProfilingMeta);
}

size_t vpux::VPUMI37XX::ProfilingMetadataOp::getAlignmentRequirements() {
    return alignof(ProfilingFB::ProfilingMeta);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ProfilingMetadataOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ProfilingMetadataOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ProfilingMetadataOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
