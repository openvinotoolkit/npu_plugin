//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/dma.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;
namespace {

// remove barrier consumers and/or producers which are controlled
// by FIFO dependency. DMA[{FIFO}]
/*
    DMA[0] DMA[0] DMA[1]       DMA[0] DMA[1]
        \    |    /               \   /
            Bar           =>       Bar
        /    |    \               /   \
    DMA[0] DMA[0] DMA[1]       DMA[0] DMA[1]
*/
void removeRedundantDependencies(mlir::func::FuncOp func, BarrierInfo& barrierInfo) {
    const auto findRedundantDependencies = [&](const BarrierInfo::TaskSet& dependencies, bool producer = true) {
        // find dependencies to remove
        BarrierInfo::TaskSet dependenciesToRemove;
        for (auto taskIndexIter = dependencies.begin(); taskIndexIter != dependencies.end(); ++taskIndexIter) {
            for (auto nextIndex = std::next(taskIndexIter); nextIndex != dependencies.end(); ++nextIndex) {
                if (!barrierInfo.controlPathExistsBetween(*taskIndexIter, *nextIndex, false)) {
                    continue;
                }

                if (producer) {
                    dependenciesToRemove.insert(*taskIndexIter);
                } else {
                    dependenciesToRemove.insert(*nextIndex);
                }
            }
        }
        return dependenciesToRemove;
    };

    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        // find producers to remove
        const auto barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
        auto producersToRemove = findRedundantDependencies(barrierProducers);
        barrierInfo.removeProducers(barrierOp, producersToRemove);

        // find consumers to remove
        const auto barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
        auto consumersToRemove = findRedundantDependencies(barrierConsumers, false);
        barrierInfo.removeConsumers(barrierOp, consumersToRemove);
    });
}

// Remove explicit barrier dependency between DMAs
// 1) if a barrier only has DMAs using single port as its producer,
//    remove all DMAs using the same port from its consumers. DMA[{FIFO}]
/*
    DMA[0] DMA[0]       DMA[0] DMA[0]
       \   /               \   /
        Bar         =>      Bar
       /   \                 |
    DMA[0] DMA[1]          DMA[1]
*/
// 2) if a barrier only has DMAs using single port as its consumer,
//    remove all DMAs using the same port from its producers. DMA[{FIFO}]
/*
    DMA[0] DMA[1]          DMA[1]
       \   /                 |
        Bar         =>      Bar
       /   \               /   \
    DMA[0] DMA[0]       DMA[0] DMA[0]
*/

void removeExplicitDependencies(mlir::func::FuncOp func, BarrierInfo& barrierInfo) {
    const auto findExplicitDependencies = [&](const BarrierInfo::TaskSet& dependencies,
                                              const VPURT::TaskQueueType& type) {
        BarrierInfo::TaskSet dependenciesToRemove;
        for (auto& taskInd : dependencies) {
            if (type == VPURT::getTaskQueueType(barrierInfo.getTaskOpAtIndex(taskInd), false)) {
                dependenciesToRemove.insert(taskInd);
            }
        }
        return dependenciesToRemove;
    };

    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        const auto barrierProducers = barrierInfo.getBarrierProducers(barrierOp);

        // try to optimize consumers (1)
        auto producerTaskQueueType = barrierInfo.haveSameImplicitDependencyTaskQueueType(barrierProducers);
        if (producerTaskQueueType.hasValue()) {
            // barrier produced by tasks with same type
            auto consumersToRemove = findExplicitDependencies(barrierInfo.getBarrierConsumers(barrierOp),
                                                              producerTaskQueueType.getValue());
            // remove consumers
            barrierInfo.removeConsumers(barrierOp, consumersToRemove);
        }

        // try to optimize producers (2)
        const auto barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
        auto consumerTaskQueueType = barrierInfo.haveSameImplicitDependencyTaskQueueType(barrierConsumers);
        if (consumerTaskQueueType.hasValue() || barrierConsumers.empty()) {
            // barrier consumed by tasks with same type
            BarrierInfo::TaskSet producersToRemove;
            // find producers to remove
            if (barrierConsumers.empty()) {
                producersToRemove = barrierProducers;
            } else {
                producersToRemove = findExplicitDependencies(barrierProducers, consumerTaskQueueType.getValue());
            }

            // remove producers
            barrierInfo.removeProducers(barrierOp, producersToRemove);
        }

        VPUX_THROW_WHEN(
                barrierInfo.getBarrierConsumers(barrierOp).empty() ^ barrierInfo.getBarrierProducers(barrierOp).empty(),
                "Invalid optimization : Only barrier {0} became empty for barrier '{1}'",
                barrierProducers.empty() ? "producers" : "consumers", barrierOp);
    });
}

// Merge barriers using FIFO order. DMA-{IR-order}
// DMA-0 and DMA-1 are before DMA-2 and DMA-3 in FIFO
/*
    DMA-0 DMA-1      DMA-0 DMA-1
      |    |            \  /
    Bar0  Bar1   =>      Bar
      |    |            /   \
    DMA-2 DMA-3      DMA-2 DMA-3
*/

void mergeBarriers(BarrierInfo& barrierInfo, ArrayRef<BarrierInfo::TaskSet> origWaitBarriersMap) {
    // Merge barriers if possible
    const auto barrierNum = barrierInfo.getNumOfVirtualBarriers();
    for (size_t barrierInd = 0; barrierInd < barrierNum; ++barrierInd) {
        auto barrierProducersA = barrierInfo.getBarrierProducers(barrierInd);
        if (barrierProducersA.empty()) {
            continue;
        }
        auto barrierConsumersA = barrierInfo.getBarrierConsumers(barrierInd);
        if (barrierConsumersA.empty()) {
            continue;
        }
        for (auto nextBarrierInd = barrierInd + 1; nextBarrierInd < barrierNum; ++nextBarrierInd) {
            auto barrierProducersB = barrierInfo.getBarrierProducers(nextBarrierInd);
            if (barrierProducersB.empty()) {
                continue;
            }
            auto barrierConsumersB = barrierInfo.getBarrierConsumers(nextBarrierInd);
            if (barrierConsumersB.empty()) {
                continue;
            }
            if (!barrierInfo.canBarriersBeMerged(barrierProducersA, barrierConsumersA, barrierProducersB,
                                                 barrierConsumersB, origWaitBarriersMap)) {
                continue;
            }

            // need to update barriers
            barrierInfo.addProducers(barrierInd, barrierProducersB);
            barrierInfo.addConsumers(barrierInd, barrierConsumersB);
            barrierInfo.resetBarrier(nextBarrierInd);
            llvm::set_union(barrierProducersA, barrierProducersB);
            llvm::set_union(barrierConsumersA, barrierConsumersB);
        }
    }
}

void postProcessBarrierOps(mlir::func::FuncOp func) {
    // move barriers to top and erase unused
    auto barrierOps = to_small_vector(func.getOps<VPURT::DeclareVirtualBarrierOp>());
    auto& block = func.getBody().front();
    VPURT::DeclareVirtualBarrierOp prevBarrier = nullptr;
    for (auto& barrierOp : barrierOps) {
        if (barrierOp.barrier().use_empty()) {
            barrierOp->erase();
            continue;
        }
        if (prevBarrier != nullptr) {
            barrierOp->moveAfter(prevBarrier);
        } else {
            barrierOp->moveBefore(&block, block.begin());
        }
        prevBarrier = barrierOp;
    }
}

//
//  DMABarrierOptimizationPass
//

class DMABarrierOptimizationPass final : public VPUIP::DMABarrierOptimizationBase<DMABarrierOptimizationPass> {
public:
    explicit DMABarrierOptimizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DMABarrierOptimizationPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();
    // Build the control relationship between any two task op. Note that the relationship includes the dependences by
    // the barriers as well as the implicit dependence by FIFO
    barrierInfo.buildTaskControllMap();

    // get original wait barrier map
    const auto origWaitBarriersMap = barrierInfo.getWaitBarriersMap();

    // DMA operation in the same FIFO do not require a barrier between them
    // optimize dependencies between DMA tasks in the same FIFO
    removeRedundantDependencies(func, barrierInfo);
    removeExplicitDependencies(func, barrierInfo);
    mergeBarriers(barrierInfo, origWaitBarriersMap);
    removeRedundantDependencies(func, barrierInfo);

    barrierInfo.orderBarriers();
    barrierInfo.updateIR();
    barrierInfo.clearAttributes();

    postProcessBarrierOps(func);
}

}  // namespace

//
// createDMABarrierOptimizationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMABarrierOptimizationPass(Logger log) {
    return std::make_unique<DMABarrierOptimizationPass>(log);
}
