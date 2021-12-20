//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/VPURT/task.hpp"

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/format.hpp"

using namespace vpux;

VPUIP::TaskOpInterface vpux::VPURT::TaskOp::getInnerTaskOp() {
    return mlir::cast<VPUIP::TaskOpInterface>(body().front().front());
}

void vpux::VPURT::TaskOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers) {
    build(odsBuilder, odsState, nullptr, waitBarriers, updateBarriers);
}

VPUIP::BlobWriter::SpecificTask vpux::VPURT::TaskOp::serialize(VPUIP::BlobWriter& writer) {
    auto task = getInnerTaskOp();
    writer.setAliasForSerializedTensors(task);
    return task.serialize(writer);
}

VPU::ExecutorKind vpux::VPURT::TaskOp::getExecutorKind() {
    return getInnerTaskOp().getExecutorKind();
}

mlir::LogicalResult vpux::VPURT::verifyTaskOp(TaskOp task) {
    if (task.body().getBlocks().size() != 1) {
        return errorAt(task, "The task body should contain excatly one block");
    }

    if (task.body().front().getOperations().size() != 1) {
        return errorAt(task, "The task body should contain exactly one operation");
    }

    auto& innerOp = task.body().front().front();
    if (!mlir::isa<VPUIP::TaskOpInterface>(innerOp)) {
        return errorAt(task, "The task body should contain VPUIP Task operation");
    }
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

    auto bodyEffects = mlir::cast<mlir::MemoryEffectOpInterface>(getInnerTaskOp().getOperation());
    bodyEffects.getEffects(effects);
}
