//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/host_parsing/host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

//
// ActKernelInvocationOp
//

void vpux::VPUIPRegMapped::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    host_parsing::ActKernelInvocationWrapper actKernelInvo;
    memset(reinterpret_cast<void*>(&actKernelInvo), 0, sizeof(actKernelInvo));

    auto index = range_index().getType().cast<VPUIPRegMapped::IndexType>();

    actKernelInvo.kRangeIndex_ = index.getValue();
    actKernelInvo.tile_ = tile();
    actKernelInvo.start_after_ = start_after();
    actKernelInvo.clean_after_ = clean_after();
    actKernelInvo.kInvo_.barriers_.group = 0;
    actKernelInvo.kInvo_.barriers_.mask = 0;

    uint64_t cons_mask = 0;
    for (auto waitBarrier : waitBarriers()) {
        if (auto op = mlir::dyn_cast_or_null<VPUIPRegMapped::ConfigureBarrierOp>(waitBarrier.getDefiningOp())) {
            cons_mask |= static_cast<uint64_t>(1) << op.id();
        }
    }
    uint64_t prod_mask = 0;
    for (auto updateBarrier : updateBarriers()) {
        if (auto op = mlir::dyn_cast_or_null<VPUIPRegMapped::ConfigureBarrierOp>(updateBarrier.getDefiningOp())) {
            prod_mask |= static_cast<uint64_t>(1) << op.id();
        }
    }

    actKernelInvo.kInvo_.barriers_.consumer_mask = cons_mask;
    actKernelInvo.kInvo_.barriers_.producer_mask = prod_mask;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelInvo);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ActKernelInvocationOp::getBinarySize() {
    return sizeof(host_parsing::ActKernelInvocationWrapper);
}
