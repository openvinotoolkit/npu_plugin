//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUIPRegMapped::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                        vpux::ELF::NetworkMetadata& metadata) {
    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&metadata);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

void vpux::VPUIPRegMapped::NetworkMetadataOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
    VPUX_THROW("ERROR");
}

size_t vpux::VPUIPRegMapped::NetworkMetadataOp::getBinarySize() {
    return sizeof(ELF::NetworkMetadata);
}
