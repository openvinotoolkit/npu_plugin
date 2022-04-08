//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuBarrierCountConfig barrier;

    barrier.next_same_id_ = next_same_id();
    barrier.consumer_count_ = consumer_count().getValueOr(0);
    barrier.producer_count_ = producer_count().getValueOr(0);
    barrier.real_id_ = id();

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barrier);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ConfigureBarrierOp::getBinarySize() {
    return sizeof(nn_public::VpuBarrierCountConfig);
}

size_t vpux::VPUIPRegMapped::ConfigureBarrierOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuBarrierCountConfig);
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::ConfigureBarrierOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ConfigureBarrierOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ConfigureBarrierOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}
