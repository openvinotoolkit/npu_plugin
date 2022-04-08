//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/mem_size.hpp"

using namespace vpux;

namespace vpux {
namespace VPUIPRegMapped {

constexpr uint32_t defaultTotalStackSize = (16_KB).to<vpux::Byte>().count();
constexpr uint32_t defaultActRtCodeSectionSize = (1_MB).to<vpux::Byte>().count();
constexpr uint32_t defaultActRtEntry = 0x1C000000;

}  // namespace VPUIPRegMapped
}  // namespace vpux

//
// MappedInferenceOp
//

void vpux::VPUIPRegMapped::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuMappedInference mi;
    memset(reinterpret_cast<void*>(&mi), 0, getBinarySize());

    mi.vpu_nnrt_api_ver = VPU_NNRT_API_VER;

    auto dmaCountVec = parseIntArrayAttr<int64_t>(dmaCount());
    VPUX_THROW_WHEN(dmaCountVec.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA lists");
    for (size_t listIdx = 0; listIdx < dmaCountVec.size(); ++listIdx) {
        mi.dma_tasks[listIdx].count = dmaCountVec[listIdx];
    }
    mi.invariants.count = invariantCount();
    mi.variants.count = variantCount();
    mi.act_kernel_ranges.count = actKernelRangesCount();
    mi.act_kernel_invocations.count = actKernelInvocationsCount();
    mi.barrier_configs.count = barrierCount();

    if (mi.act_kernel_invocations.count) {
        mi.shv_rt_configs.stack_size = VPUIPRegMapped::defaultTotalStackSize;
        mi.shv_rt_configs.use_schedule_embedded_rt = false;
        mi.shv_rt_configs.code_window_buffer_size = VPUIPRegMapped::defaultActRtCodeSectionSize;
        mi.shv_rt_configs.runtime_version = 0;
        mi.shv_rt_configs.runtime_entry = VPUIPRegMapped::defaultActRtEntry;

        auto actShvRtOp = mlir::dyn_cast<VPUIPRegMapped::ActShaveRtOp>(actShaveRt().getDefiningOp());
        if (actShvRtOp) {
            mi.shv_rt_configs.use_schedule_embedded_rt = true;
            mi.shv_rt_configs.code_window_buffer_size = actShvRtOp.getBinarySize();
            mi.shv_rt_configs.runtime_version = actShvRtOp.getVersion();
            mi.shv_rt_configs.runtime_entry = actShvRtOp.getKernelEntry();
        }

        mi.shv_rt_configs.stack_size = 0;
        for (auto actShaveStack : actShaveStacks()) {
            mi.shv_rt_configs.stack_size +=
                    actShaveStack.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
        }
    }

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&mi);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUIPRegMapped::MappedInferenceOp::getBinarySize() {
    return sizeof(nn_public::VpuMappedInference);
}

size_t vpux::VPUIPRegMapped::MappedInferenceOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuMappedInference);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::MappedInferenceOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

vpux::ELF::SectionFlagsAttr vpux::VPUIPRegMapped::MappedInferenceOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUIPRegMapped::MappedInferenceOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

mlir::FailureOr<uint64_t> vpux::VPUIPRegMapped::MappedInferenceOp::getOffsetOfWithinOperation(mlir::Value val) {
    if (val == barrierTasks()) {
        return offsetof(nn_public::VpuMappedInference, barrier_configs) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuBarrierCountConfig>, address);
    } else if (val == actKernelInvocations()) {
        return offsetof(nn_public::VpuMappedInference, act_kernel_invocations) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelInvocation>, address);
    } else if (val == actKernelRanges()) {
        return offsetof(nn_public::VpuMappedInference, act_kernel_ranges) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuActKernelRange>, address);
    } else if (val == variantTasks()) {
        return offsetof(nn_public::VpuMappedInference, variants) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUVariant>, address);
    } else if (val == invariantTasks()) {
        return offsetof(nn_public::VpuMappedInference, invariants) +
               offsetof(nn_public::VpuTaskReference<nn_public::VpuDPUInvariant>, address);
    } else if (val == actShaveRt()) {
        return offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
               offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base);
    }
    for (auto actShaveStack : actShaveStacks() | indexed) {
        if (val == actShaveStack.value()) {
            const auto index = actShaveStack.index();
            return offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
                   offsetof(nn_public::VpuNNShaveRuntimeConfigs, stack_frames) +
                   index * sizeof(nn_public::VpuNNShaveRuntimeConfigs::stack_frames[0]);
        }
    }
    for (auto listHead : dmaTasks()) {
        auto listIdx = mlir::cast<VPUIPRegMapped::NNDMAOp>(listHead.getDefiningOp()).port();
        if (listHead == val) {
            return offsetof(nn_public::VpuMappedInference, dma_tasks) +
                   (sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>) * listIdx) +
                   offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address);
        }
    }

    return 0;
}
