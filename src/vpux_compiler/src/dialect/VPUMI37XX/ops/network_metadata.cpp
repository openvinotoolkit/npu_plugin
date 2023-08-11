//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

using namespace vpux;

//
//  NetworkMetadataOp
//

void vpux::VPUMI37XX::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                   elf::NetworkMetadata& metadata) {
    auto operation = getOperation();

    auto nBarrs = VPUIP::getNumAvailableBarriers(operation);
    metadata.resource_requirements.nn_barriers_ = nBarrs;
    metadata.resource_requirements.nn_slice_count_ =
            VPUIP::getNumClusterUsed(operation->getParentOfType<mlir::ModuleOp>());

    metadata.resource_requirements.ddr_scratch_length_ = checked_cast<uint32_t>(
            IE::getAvailableMemory(operation->getParentOfType<mlir::ModuleOp>(), vpux::VPU::MemoryKind::DDR)
                    .byteSize());

    metadata.resource_requirements.nn_slice_length_ = checked_cast<uint32_t>(
            IE::getAvailableMemory(operation->getParentOfType<mlir::ModuleOp>(), vpux::VPU::MemoryKind::CMX_NN)
                    .byteSize());

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&metadata);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

void vpux::VPUMI37XX::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
    VPUX_THROW("ERROR");
}

size_t vpux::VPUMI37XX::NetworkMetadataOp::getBinarySize() {
    return sizeof(elf::NetworkMetadata);
}

size_t vpux::VPUMI37XX::NetworkMetadataOp::getAlignmentRequirements() {
    return alignof(elf::NetworkMetadata);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::NetworkMetadataOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::NetworkMetadataOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::NetworkMetadataOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
