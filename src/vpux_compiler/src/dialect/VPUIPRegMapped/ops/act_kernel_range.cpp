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
// ActKernelRangeOp
//

void vpux::VPUIPRegMapped::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {

    auto kernel_text = kernel_text_index();
    auto kernel_args = kernel_args_index();
    auto kernel_entry = kernel_entry_index();

    auto kernel_text_op = llvm::dyn_cast<VPUIPRegMapped::DeclareKernelTextOp>(kernel_text.getDefiningOp());
    auto kernel_args_op = llvm::dyn_cast<VPUIPRegMapped::DeclareKernelArgsOp>(kernel_args.getDefiningOp());
    auto kernel_entry_op = llvm::dyn_cast<VPUIPRegMapped::DeclareKernelEntryOp>(kernel_entry.getDefiningOp());


    auto kernel_text_size = kernel_text_op.getBinarySize();
    auto kernel_args_size = kernel_args_op.getBinarySize();
    auto kernel_entryy = kernel_entry_op.getKernelEntry();


    uint32_t invo_count = 0;
    auto op = this;
    auto uses = op->getResult().getUses();
    for(auto use = uses.begin(); use != uses.end(); use++){
        if(strcmp(use.getUser()->getName().getStringRef().data(), "VPUIPRegMapped.ActKernelInvocation") == 0)
            invo_count++;
    }


    host_parsing::ActKernelRangeWrapper actKernelRange;

    memset(reinterpret_cast<void*>(&actKernelRange), 0, sizeof(actKernelRange));

    actKernelRange.kInvoCount_ = invo_count;
    actKernelRange.kRange_.type_ = host_parsing::WL_KERNEL;
    actKernelRange.kRange_.kernelEntry_ = kernel_entryy;
    actKernelRange.kRange_.codeSize_ = kernel_text_size;
    actKernelRange.kRange_.dataSecSize_ = kernel_args_size;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelRange);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ActKernelRangeOp::getBinarySize() {

    return sizeof(host_parsing::ActKernelRangeWrapper);
}
