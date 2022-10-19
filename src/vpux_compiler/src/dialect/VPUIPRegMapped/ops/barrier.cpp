//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    host_parsing::BarrierWrapper barrier;

    barrier.next_same_id = next_same_id();
    barrier.real_id = id();
    barrier.consumer_count = consumer_count().getValueOr(0);
    barrier.producer_count = producer_count().getValueOr(0);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&barrier);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ConfigureBarrierOp::getBinarySize() {
    return sizeof(host_parsing::BarrierWrapper);
}
