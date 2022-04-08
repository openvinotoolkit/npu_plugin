//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/compiler/dialect/VPUIPRegMapped/utils.hpp"

using namespace vpux;

//
// ActKernelInvocationOp
//

void vpux::VPUIPRegMapped::ActKernelInvocationOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuActKernelInvocation actKernelInvocation;
    memset(reinterpret_cast<void*>(&actKernelInvocation), 0, sizeof(actKernelInvocation));

    auto rangeIndex = range_index().getType().cast<VPUIPRegMapped::IndexType>();
    auto invoIndex = getType().cast<VPUIPRegMapped::IndexType>();

    actKernelInvocation.range = rangeIndex.getValue();
    actKernelInvocation.kernel_range_index = rangeIndex.getValue();
    actKernelInvocation.invo_tile = tile();
    actKernelInvocation.barriers_sched.start_after_ = start_after();
    actKernelInvocation.barriers_sched.clean_after_ = clean_after();
    actKernelInvocation.invo_index = invoIndex.getValue();
    actKernelInvocation.barriers.wait_mask_ = VPUIPRegMapped::computeMask(waitBarriers());
    actKernelInvocation.barriers.post_mask_ = VPUIPRegMapped::computeMask(updateBarriers());

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

size_t vpux::VPUIPRegMapped::ActKernelInvocationOp::getBinarySize() {
    return sizeof(nn_public::VpuActKernelInvocation);
}

size_t vpux::VPUIPRegMapped::ActKernelInvocationOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuActKernelInvocation);
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::ActKernelInvocationOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ActKernelInvocationOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::ActKernelInvocationOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

mlir::FailureOr<uint64_t> vpux::VPUIPRegMapped::ActKernelInvocationOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == range_index()) {
        return offsetof(nn_public::VpuActKernelInvocation, range);
    }

    return mlir::failure();
}
