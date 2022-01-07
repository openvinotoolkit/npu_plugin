//
// Copyright 2020 Intel Corporation.
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

#include <host_parsed_inference.h>
#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include <iostream>

using namespace vpux;

//
// ActKernelInvocationOp
//

void vpux::VPUIPRegMapped::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {

    host_parsing::ActKernelInvocationWrapper actKernelInvo;
    memset(reinterpret_cast<void*>(&actKernelInvo), 0, sizeof(actKernelInvo));

    auto index = range_index().getType().cast<VPUIPRegMapped::IndexType>();

    actKernelInvo.kRangeIndex_ =  index.getValue();
    actKernelInvo.tile_ = tile();
    actKernelInvo.start_after_ = start_after();
    actKernelInvo.clean_after_ = clean_after();
    actKernelInvo.kInvo_.barriers_.group = 0;
    actKernelInvo.kInvo_.barriers_.mask = 0;

    uint64_t cons_mask = 0;
    // for (auto waitBarrier : waitBarriers()) {
    auto wait_op = llvm::dyn_cast<VPUIPRegMapped::ConfigureBarrierOp>(waitBarriers().getDefiningOp());
    cons_mask |= 1 << wait_op.id();
    // }
    uint64_t prod_mask = 0;
    // for (auto updateBarrier : updateBarriers()) {
    auto update_op = llvm::dyn_cast<VPUIPRegMapped::ConfigureBarrierOp>(updateBarriers().getDefiningOp());
    prod_mask |= 1 << update_op.id();
    // }

    // printf("\n\nPROD MASK: %lu \nCONS MASK: %lu\n\n", prod_mask, cons_mask);

    actKernelInvo.kInvo_.barriers_.consumer_mask = cons_mask;
    actKernelInvo.kInvo_.barriers_.producer_mask = prod_mask;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelInvo);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ActKernelInvocationOp::getBinarySize() {
    return sizeof(host_parsing::ActKernelInvocationWrapper);
}
