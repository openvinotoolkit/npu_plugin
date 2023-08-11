//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPU37XX/api/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

using namespace vpux;

//
// ActKernelRangeOp
//

void vpux::VPUMI37XX::ActKernelRangeOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto kernel_text_value = kernel_text_index();
    auto kernel_args_value = kernel_args_index();
    auto kernel_entry_value = kernel_entry_index();

    auto kernel_text_op = kernel_text_value.getDefiningOp<VPUMI37XX::DeclareKernelTextOp>();
    auto kernel_args_op = kernel_args_value.getDefiningOp<VPUMI37XX::DeclareKernelArgsOp>();
    auto kernel_entry_op = kernel_entry_value.getDefiningOp<VPUMI37XX::DeclareKernelEntryOp>();

    auto kernel_text_size = kernel_text_op.getBinarySize();
    auto kernel_args_size = kernel_args_op.getBinarySize();
    auto kernel_entry = kernel_entry_op.getKernelEntry();

    auto op = this;
    auto uses = op->getResult().getUses();

    uint32_t invo_count = 0;

    for (auto use = uses.begin(); use != uses.end(); use++) {
        if (mlir::isa<VPUMI37XX::ActKernelInvocationOp>(use.getUser()))
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

size_t vpux::VPUMI37XX::ActKernelRangeOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelRange);
}

size_t vpux::VPUMI37XX::ActKernelRangeOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelRange);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ActKernelRangeOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ActKernelRangeOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ActKernelRangeOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

mlir::FailureOr<uint64_t> vpux::VPUMI37XX::ActKernelRangeOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == kernel_text_index()) {
        return offsetof(nn_public::VpuActKernelRange, text_window_base);
    } else if (val == kernel_args_index()) {
        // kernel_args operand needs to be moved to ActKernelInvocationOp (there is where it gets relocated)
        return mlir::failure();
    }

    return mlir::failure();
}

vpux::VPURegMapped::TaskType vpux::VPUMI37XX::ActKernelRangeOp::getTaskType() {
    return vpux::VPURegMapped::TaskType::ActKernelRange;
}
