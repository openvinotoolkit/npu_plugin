//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPU37XX/api/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUMI37XX::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuBarrierCountConfig barrier;

    barrier.next_same_id_ = next_same_id();
    barrier.consumer_count_ = consumer_count().value_or(0);
    barrier.producer_count_ = producer_count().value_or(0);
    barrier.real_id_ = id();

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barrier);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::ConfigureBarrierOp::getBinarySize() {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::VPUMI37XX::ConfigureBarrierOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuBarrierCountConfig);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ConfigureBarrierOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ConfigureBarrierOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ConfigureBarrierOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}
