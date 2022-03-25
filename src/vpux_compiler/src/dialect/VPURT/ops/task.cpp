//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/task.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

mlir::Operation* vpux::VPURT::TaskOp::getInnerTaskOp() {
    return &body().front().front();
}

void vpux::VPURT::TaskOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers) {
    build(odsBuilder, odsState, nullptr, waitBarriers, updateBarriers);
}

VPUIP::BlobWriter::SpecificTask vpux::VPURT::TaskOp::serialize(VPUIP::BlobWriter& writer) {
    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(getInnerTaskOp());
    VPUX_THROW_UNLESS(task != nullptr, "Inner task  does not implement TaskOpInterface");

    writer.setAliasForSerializedTensors(task);
    return task.serialize(writer);
}

VPU::ExecutorKind vpux::VPURT::TaskOp::getExecutorKind() {
    auto innerTaskOp = getInnerTaskOp();
    if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerTaskOp)) {
        innerTaskOp = clusterTilingOp.getInnerTaskOp();
    }
    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(innerTaskOp);
    VPUX_THROW_UNLESS(task != nullptr, "Inner task  does not implement TaskOpInterface");

    return task.getExecutorKind();
}

mlir::LogicalResult vpux::VPURT::verifyTaskOp(TaskOp task) {
    if (task.body().getBlocks().size() != 1) {
        return errorAt(task, "The task body should contain exactly one block");
    }

    auto numOps = task.body().front().getOperations().size();
    if (numOps != 1) {
        return errorAt(task, "The task body should contain exactly one operation. Got: {0}", numOps);
    }

    auto& innerOp = task.body().front().front();
    if (!mlir::isa<mlir::MemoryEffectOpInterface>(innerOp)) {
        return errorAt(task, "The task body should contain operation with memory effects");
    }

    if (task.isTrailingSWLayer()) {
        for (auto updateBarrier : task.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                auto depTask = mlir::dyn_cast<VPURT::TaskOp>(depOp);

                if (depTask == nullptr) {
                    return errorAt(task, "Trailing UPA Task has non-SW dependency : '{0}'", depOp->getLoc());
                }

                if (depTask.getExecutorKind() != VPU::ExecutorKind::SHAVE_UPA) {
                    return errorAt(task, "Trailing UPA Task has non-SW dependency : '{0}'", depOp->getLoc());
                }
            }
        }
    }
    return mlir::success();
}

void vpux::VPURT::TaskOp::getEffects(SmallVectorImpl<MemoryEffect>& effects) {
    for (const auto waitBarrier : waitBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), waitBarrier, VPURT::BarrierResource::get());
    }

    for (const auto updateBarrier : updateBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), updateBarrier, VPURT::BarrierResource::get());
    }

    auto bodyEffects = mlir::cast<mlir::MemoryEffectOpInterface>(getInnerTaskOp());
    bodyEffects.getEffects(effects);
}
