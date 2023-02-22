//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/vpu_nnrt_api.h"
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

    nn_public::VpuActKernelRange actKernelRange;

    memset(reinterpret_cast<void*>(&actKernelRange), 0, sizeof(actKernelRange));

    actKernelRange.type = nn_public::VpuActWLType::WL_KERNEL;
    actKernelRange.kernel_entry = kernel_entry;
    // text_window_base not used
    actKernelRange.code_size = kernel_text_size;
    actKernelRange.data_sec_size = kernel_args_size;
    actKernelRange.kernel_invo_count = invo_count;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelRange);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::ActKernelRangeOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelRange);
}

size_t vpux::VPUIPRegMapped::ActKernelRangeOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelRange);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ActKernelRangeOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ActKernelRangeOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::ActKernelRangeOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

mlir::FailureOr<uint64_t> vpux::VPUIPRegMapped::ActKernelRangeOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == kernel_text_index()) {
        return offsetof(nn_public::VpuActKernelRange, text_window_base);
    } else if (val == kernel_args_index()) {
        // kernel_args operand needs to be moved to ActKernelInvocationOp (there is where it gets relocated)
        return mlir::failure();
    }

    return mlir::failure();
}
