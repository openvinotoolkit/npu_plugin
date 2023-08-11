//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <llvm/ADT/None.h>
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/dialect/VPU37XX/api/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_writer.hpp"

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"

namespace {
#include "external/runtime_dpu_parser_imports.cpp.inc"
}

using namespace vpux;

#define SLICE_LENGTH \
    (2 * 1024 * 1024)  // TODO: E#54008	Don't hardcode this . Check if we can do it via relocation from CMX symbol size?

void vpux::VPUMI37XX::DPUInvariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    vpux::Logger logger("DPU_Serializer", vpux::LogLevel::Debug);

    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_WHEN(parentModule == nullptr, "Could not get the parent module for DPU Invariant OP");

    VPUIP::BlobWriter writer(logger, VPU::getArch(parentModule));

    nn_public::VpuDPUInvariant taskWrapper{};
    nn_public::VpuDPUInvariantRegisters& registers = taskWrapper.registers_;

    memset(reinterpret_cast<void*>(&registers), 0, sizeof(registers));

    SetupInput(writer, *this, registers);
    SetupWeights(writer, *this, registers);
    SetupKernel(*this, registers);
    SetupOutput(writer, *this, registers);

    auto taskType = task_type();
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        SetupInvariant_Convolution(writer, *this, registers);
        break;
    case VPUIP::NCETaskType::MAXPOOL:
        SetupInvariant_MaxPool(writer, *this, registers);
        break;
    case VPUIP::NCETaskType::AVEPOOL:
        registers.elops_wload.elops_wload_bf.pool_wt_rd_dis = 1;
        SetupInvariant_DwConvolution(writer, *this, registers);
        break;
    case VPUIP::NCETaskType::DWCONV:
        SetupInvariant_DwConvolution(writer, *this, registers);
        break;
    case VPUIP::NCETaskType::ELTWISE:
        SetupInvariant_Eltwise(writer, *this, registers);
        break;
    default:
        VPUX_THROW("Unsupported task type {0}", taskType);
        break;
    }

    Setup_PPE(writer, *this, registers);

    // imported from update_invariant
    {
        // TODO: E#54009 hardcoded and directly copied from POC runtime...
        registers.base_offset_a = 0x200;
        registers.base_offset_b = 0x602;

        registers.se_sp_addr[1].se_addr = ((1 * SLICE_LENGTH) >> 4);
        registers.se_sp_addr[2].se_addr = ((1 * SLICE_LENGTH) >> 4);
        registers.se_sp_addr[3].se_addr = ((1 * SLICE_LENGTH) >> 4);

        if (task_type() == VPUIP::NCETaskType::ELTWISE) {
            auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(weights().getDefiningOp()).byteOffset();
            auto actOffs = mlir::cast<VPURT::DeclareBufferOp>(input().getDefiningOp()).byteOffset();

            registers.tensor_start = std::max(static_cast<int64_t>(0), (actOffs - weightsOffs) >> 4);
            registers.weight_start = std::max(static_cast<int64_t>(0), (weightsOffs - actOffs) >> 4);
        }

        llvm::SmallVector<mlir::Value> outputs(output_buffs());
        llvm::sort(outputs.begin(), outputs.end(), [](mlir::Value lhs, mlir::Value rhs) {
            auto lhsIdx = lhs.getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
            auto rhsIdx = rhs.getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
            return lhsIdx < rhsIdx;
        });

        auto firstIndex = outputs[0].getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
        for (size_t idx = 1; idx < outputs.size(); ++idx) {
            auto outIdx = outputs[idx].getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().value_or(0);
            if (outIdx == firstIndex) {
                continue;
            }

            registers.odu_cast[idx - 1].odu_cast_bf.cast_enable = 1;
            registers.odu_cast[idx - 1].odu_cast_bf.cast_offset = ((outIdx - firstIndex) * SLICE_LENGTH) >> 4;
        }
    }

    auto bufOp = input().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(bufOp, "Parent of DPU op is not a declareBufferOp");

    taskWrapper.cluster_ = bufOp.getNonEmptySectionIndex()[0];  // TODO:E#54007 use from memref and don't assume index 0

    taskWrapper.is_cont_conv_ = is_continued().value_or(false);

    if (auto profBuffer = profiling_data()) {
        auto profBufOp = profBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        taskWrapper.hwp_cmx_base_offset_ = profBufOp.byteOffset();
    }

    // Barriers setup
    taskWrapper.barriers_.wait_mask_ = VPUMI37XX::computeMask(waitBarriers());
    taskWrapper.barriers_.post_mask_ = VPUMI37XX::computeMask(updateBarriers());

    taskWrapper.barriers_.group_ = 0;  // will not support barrier grouping for now
    taskWrapper.barriers_.mask_ = 0;

    taskWrapper.barriers_sched_.start_after_ = start_after();
    taskWrapper.barriers_sched_.clean_after_ = clean_after();

    uint32_t variant_count = 0;
    for (auto& user : getResult().getUses()) {
        auto owner = user.getOwner();
        if (mlir::dyn_cast_or_null<VPUMI37XX::DPUVariantOp>(owner)) {
            variant_count++;
        }
    }

    taskWrapper.variant_count_ = variant_count;

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&taskWrapper);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::DPUInvariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUInvariant);
}

size_t vpux::VPUMI37XX::DPUInvariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUInvariant);
}

mlir::FailureOr<uint64_t> vpux::VPUMI37XX::DPUInvariantOp::getOffsetOfWithinOperation(mlir::Value /*value*/) {
    VPUX_THROW("OffsetOf not supported for DPUInvariant");
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::DPUInvariantOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::DPUInvariantOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_DPU);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::DPUInvariantOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::VPURegMapped::TaskType vpux::VPUMI37XX::DPUInvariantOp::getTaskType() {
    return vpux::VPURegMapped::TaskType::DPUInvariant;
}

void vpux::VPUMI37XX::DPUVariantOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    vpux::Logger logger("DPU_Serializer", vpux::LogLevel::Debug);

    auto parentModule = getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_WHEN(parentModule == nullptr, "Could not get the parent module for DPU Invariant OP");

    VPUIP::BlobWriter writer(logger, VPU::getArch(parentModule));

    auto invariant = Invariant().getDefiningOp<vpux::VPUMI37XX::DPUInvariantOp>();
    VPUX_THROW_WHEN(invariant == nullptr, "variant not pointing to an invariant");

    nn_public::VpuDPUVariant taskWrapper{};
    memset(reinterpret_cast<void*>(&taskWrapper), 0, sizeof(taskWrapper));

    parseVariant(*this, invariant, taskWrapper, writer);

    taskWrapper.invariant_index_ = invariant.getType().getValue();
    taskWrapper.invariant_ = invariant.getType().getValue();
    taskWrapper.cluster_ = 0;

    auto opType = invariant.task_type();
    if (opType == VPUIP::NCETaskType::ELTWISE) {
        auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(invariant.weights().getDefiningOp()).byteOffset();
        auto actOffs = mlir::cast<VPURT::DeclareBufferOp>(invariant.input().getDefiningOp()).byteOffset();

        auto weight_start = std::max(static_cast<int64_t>(0), (weightsOffs - actOffs) >> 4);
        taskWrapper.weight_table_offset_ += weight_start;
    }

    // in case of disabled profiling runtime expects default (empty) value
    // to be -1 instead of 0
    taskWrapper.wload_id_ = workload_id().value_or(-1);

    auto ptrCharTmp = reinterpret_cast<uint8_t*>(&taskWrapper);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::VPUMI37XX::DPUVariantOp::getBinarySize() {
    return sizeof(nn_public::VpuDPUVariant);
}

size_t vpux::VPUMI37XX::DPUVariantOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuDPUVariant);
}

mlir::FailureOr<uint64_t> vpux::VPUMI37XX::DPUVariantOp::getOffsetOfWithinOperation(mlir::Value value) {
    if (value == Invariant()) {
        return offsetof(nn_public::VpuDPUVariant, invariant_);
    }

    return mlir::failure();
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::DPUVariantOp::getAccessingProcs() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR | ELF::SectionFlagsAttr::VPU_SHF_PROC_DMA);
}

vpux::ELF::SectionFlagsAttr vpux::VPUMI37XX::DPUVariantOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_DPU);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::DPUVariantOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}

vpux::VPURegMapped::TaskType vpux::VPUMI37XX::DPUVariantOp::getTaskType() {
    return vpux::VPURegMapped::TaskType::DPUVariant;
}
