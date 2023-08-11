//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/task.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/types.hpp"

#include "vpux/compiler/utils/dma.hpp"
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

mlir::LogicalResult vpux::VPURT::TaskOp::verify() {
    const auto task = getOperation();
    if (body().getBlocks().size() != 1) {
        return errorAt(task, "The task body should contain exactly one block");
    }

    auto numOps = body().front().getOperations().size();
    if (numOps != 1) {
        return errorAt(task, "The task body should contain exactly one operation. Got: {0}", numOps);
    }

    auto& innerOp = body().front().front();
    if (!mlir::isa<mlir::MemoryEffectOpInterface>(innerOp)) {
        return errorAt(task, "The task body should contain operation with memory effects");
    }

    if (isTrailingSWLayer()) {
        for (auto updateBarrier : updateBarriers()) {
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

SmallVector<int64_t> vpux::VPURT::getDMATaskPorts(TaskOp task) {
    VPUX_THROW_UNLESS(task.getExecutorKind() == VPU::ExecutorKind::DMA_NN, "Unexpected task type find at '{0}'",
                      task->getLoc());
    SmallVector<int64_t> ports;
    auto* wrappedTaskOp = task.getInnerTaskOp();
    if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(wrappedTaskOp)) {
        auto module = clusterTilingOp->getParentOfType<mlir::ModuleOp>();
        auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
        auto dmaPortCount = dmaOp.count();

        const auto input = *clusterTilingOp.getInputs().begin();
        const auto output = *clusterTilingOp.getOutputs().begin();
        auto inputType = input.getType().cast<vpux::NDTypeInterface>();
        auto outputType = output.getType().cast<vpux::NDTypeInterface>();

        const auto distributedType = inputType.isa<VPUIP::DistributedBufferType>()
                                             ? inputType.dyn_cast<VPUIP::DistributedBufferType>()
                                             : outputType.dyn_cast<VPUIP::DistributedBufferType>();

        VPUX_THROW_UNLESS(distributedType != nullptr, "At least one of operands must have DistributedBuffer type");

        int64_t numClusters = 0;
        const auto checkSegmentedOrOverlapped = [&](vpux::NDTypeInterface type) {
            const auto distType = type.dyn_cast<VPUIP::DistributedBufferType>();
            if (distType == nullptr) {
                return false;
            }
            const auto distributionAttr = distType.getDistribution();
            VPUX_THROW_WHEN(distributionAttr == nullptr, "Failed to get distribution attribute.");
            const auto mode = distributionAttr.mode().getValue();
            if (mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED) {
                numClusters = distributionAttr.num_clusters().getInt();
                return true;
            }
            return false;
        };

        if (checkSegmentedOrOverlapped(inputType) || checkSegmentedOrOverlapped(inputType)) {
            ports.resize(std::min(numClusters, dmaPortCount));
            std::iota(ports.begin(), ports.end(), 0);
            return ports;
        }
        wrappedTaskOp = clusterTilingOp.getInnerTaskOp();
    }

    ports.push_back(vpux::getDMAPortValue(wrappedTaskOp));
    return ports;
}

Optional<SmallVector<VPURT::TaskQueueType>> vpux::VPURT::getDMATaskQueueType(TaskOp taskOp) {
    if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
        return None;
    }
    SmallVector<VPURT::TaskQueueType> queueTypes;
    auto ports = VPURT::getDMATaskPorts(taskOp);
    for (auto port : ports) {
        TaskQueueType queueType;
        queueType.type = VPU::ExecutorKind::DMA_NN;
        queueType.index = port;
        queueTypes.push_back(queueType);
    }
    return queueTypes;
}

VPURT::TaskQueueType vpux::VPURT::getTaskQueueType(TaskOp taskOp, bool ignoreIndexForNce) {
    TaskQueueType queueType;
    queueType.type = taskOp.getExecutorKind();
    if (queueType.type == VPU::ExecutorKind::NCE && !ignoreIndexForNce) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp();
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(wrappedTaskOp)) {
            wrappedTaskOp = clusterTilingOp.getInnerTaskOp();
        }
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
        VPUX_THROW_WHEN(nceTask == nullptr || nceTask.variants().getOps<VPUIP::DPUTaskOp>().empty(),
                        "Could not get DPU task");
        auto dpuTask = *(nceTask.variants().getOps<VPUIP::DPUTaskOp>().begin());
        queueType.index = dpuTask.cluster_id().value_or(0);
    } else if (queueType.type == VPU::ExecutorKind::DMA_NN) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp();
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(wrappedTaskOp)) {
            wrappedTaskOp = clusterTilingOp.getInnerTaskOp();
        }
        queueType.index = vpux::getDMAPortValue(wrappedTaskOp);
    } else {
        queueType.index = 0;
    }
    return queueType;
}
