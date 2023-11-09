//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"

#include <vpu_nnrt_api_37xx.h>

using namespace vpux;

//
// ActKernelInvocationOp
//

void vpux::VPUMI37XX::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuActKernelInvocation actKernelInvocation;
    memset(reinterpret_cast<void*>(&actKernelInvocation), 0, sizeof(actKernelInvocation));

    auto rangeIndex = getRangeIndex().getType().cast<VPURegMapped::IndexType>();
    auto invoIndex = getType().cast<VPURegMapped::IndexType>();

    actKernelInvocation.range = rangeIndex.getValue();
    actKernelInvocation.kernel_range_index = rangeIndex.getValue();
    actKernelInvocation.invo_tile = getTile();
    actKernelInvocation.barriers_sched.start_after_ = getStartAfter();
    actKernelInvocation.barriers_sched.clean_after_ = getCleanAfter();
    actKernelInvocation.invo_index = invoIndex.getValue();
    actKernelInvocation.barriers.wait_mask_ = VPUMI37XX::computeMask(getWaitBarriers());
    actKernelInvocation.barriers.post_mask_ = VPUMI37XX::computeMask(getUpdateBarriers());

    actKernelInvocation.barriers.group_ = 0;
    actKernelInvocation.barriers.mask_ = 0;

    for (uint64_t mask = actKernelInvocation.barriers.wait_mask_, group = 1; mask > 0; mask >>= 8, ++group) {
        if (mask & 0xff) {
            if (actKernelInvocation.barriers.group_ == 0) {
                actKernelInvocation.barriers.group_ = static_cast<unsigned char>(group);
                actKernelInvocation.barriers.mask_ = mask & 0xff;
            } else {
                actKernelInvocation.barriers.group_ = 0;
                actKernelInvocation.barriers.mask_ = 0;
                break;
            }
        }
    }

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&actKernelInvocation);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::ActKernelInvocationOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelInvocation);
}

size_t vpux::VPUMI37XX::ActKernelInvocationOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelInvocation);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::ActKernelInvocationOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ActKernelInvocationOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::ActKernelInvocationOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

mlir::FailureOr<uint64_t> vpux::VPUMI37XX::ActKernelInvocationOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == getRangeIndex()) {
        return offsetof(nn_public::VpuActKernelInvocation, range);
    }

    return mlir::failure();
}

vpux::VPURegMapped::TaskType vpux::VPUMI37XX::ActKernelInvocationOp::getTaskType() {
    return vpux::VPURegMapped::TaskType::ActKernelInvocation;
}
