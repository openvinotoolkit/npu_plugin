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

//
// TaskOp
//

VPUIP::BlobWriter::SpecificTask vpux::VPURT::TaskOp::serialize(VPUIP::BlobWriter& writer) {
    auto& block = op().getBlocks().front();
    VPUX_THROW_UNLESS(block.getOperations().size() == 1, "Unable to find child task in VPURT::TaskOp for serialize");
    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(block.begin());
    auto& op = *block.begin();
    writer.setAliasForSerializedTensors(&op);
    return task.serialize(writer);
}

vpux::VPUIP::TaskType vpux::VPURT::TaskOp::getTaskType() {
    auto& block = op().getBlocks().front();
    VPUX_THROW_UNLESS(block.getOperations().size() == 1, "Unable to find child task in VPURT::TaskOp for get TaskType");
    auto task = mlir::dyn_cast<VPUIP::TaskOpInterface>(block.begin());
    VPUX_THROW_UNLESS(task != nullptr, "Unable to get TaskType");
    return task.getTaskType();
}

mlir::LogicalResult vpux::VPURT::verifyTaskOp(TaskOp task) {
    if (task.isTrailingSWLayer()) {
        for (auto updateBarrier : task.updateBarriers()) {
            for (auto* depOp : updateBarrier.getUsers()) {
                auto depTask = mlir::dyn_cast<VPURT::TaskOp>(depOp);

                if (depTask == nullptr) {
                    return errorAt(task, "Trailing UPA Task has non-SW dependency : '{0}'", depOp->getLoc());
                }

                if (depTask.getTaskType() != VPUIP::TaskType::UPA) {
                    return errorAt(task, "Trailing UPA Task has non-SW dependency : '{0}'", depOp->getLoc());
                }
            }
        }
    }
    return mlir::success();
}

//
// TaskOpInterface
//

void vpux::VPURT::TaskOp::getEffects(SmallVectorImpl<MemoryEffect>& effects) {
    for (const auto waitBarrier : waitBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Read::get(), waitBarrier, VPURT::BarrierResource::get());
    }

    for (const auto updateBarrier : updateBarriers()) {
        effects.emplace_back(mlir::MemoryEffects::Write::get(), updateBarrier, VPURT::BarrierResource::get());
    }

    for (auto& op : this->op().front()) {
        if (auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
            opEffects.getEffects(effects);
        }
    }
}
