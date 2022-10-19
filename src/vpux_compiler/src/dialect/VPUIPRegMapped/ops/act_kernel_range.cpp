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
// ActKernelRangeOp
//

void vpux::VPUIPRegMapped::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel_text_value = kernel_text_index();
    auto kernel_args_value = kernel_args_index();
    auto kernel_entry_value = kernel_entry_index();

    auto kernel_text_op = kernel_text_value.getDefiningOp<VPUIPRegMapped::DeclareKernelTextOp>();
    auto kernel_args_op = kernel_args_value.getDefiningOp<VPUIPRegMapped::DeclareKernelArgsOp>();
    auto kernel_entry_op = kernel_entry_value.getDefiningOp<VPUIPRegMapped::DeclareKernelEntryOp>();

    auto kernel_text_size = kernel_text_op.getBinarySize();
    auto kernel_args_size = kernel_args_op.getBinarySize();
    auto kernel_entry = kernel_entry_op.getKernelEntry();

    auto op = this;
    auto uses = op->getResult().getUses();

    uint32_t invo_count = 0;

    for (auto use = uses.begin(); use != uses.end(); use++) {
        if (mlir::isa<VPUIPRegMapped::ActKernelInvocationOp>(use.getUser()))
            invo_count++;
    }

    host_parsing::ActKernelRangeWrapper actKernelRange;

    memset(reinterpret_cast<void*>(&actKernelRange), 0, sizeof(actKernelRange));

    actKernelRange.kInvoCount_ = invo_count;
    actKernelRange.kRange_.type_ = host_parsing::WL_KERNEL;
    actKernelRange.kRange_.kernelEntry_ = kernel_entry;
    actKernelRange.kRange_.codeSize_ = kernel_text_size;
    actKernelRange.kRange_.dataSecSize_ = kernel_args_size;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelRange);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ActKernelRangeOp::getBinarySize() {
    return sizeof(host_parsing::ActKernelRangeWrapper);
}
