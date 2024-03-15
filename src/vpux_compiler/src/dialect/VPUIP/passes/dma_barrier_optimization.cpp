//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

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
        const auto& barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
        const auto producersToRemove = findRedundantDependencies(barrierProducers);
        barrierInfo.removeProducers(barrierOp, producersToRemove);

        // find consumers to remove
        const auto& barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
        const auto consumersToRemove = findRedundantDependencies(barrierConsumers, false);
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
        const auto& barrierProducers = barrierInfo.getBarrierProducers(barrierOp);

        // try to optimize consumers (1)
        auto producerTaskQueueType = barrierInfo.haveSameImplicitDependencyTaskQueueType(barrierProducers);
        if (producerTaskQueueType.has_value()) {
            // barrier produced by tasks with same type
            auto consumersToRemove =
                    findExplicitDependencies(barrierInfo.getBarrierConsumers(barrierOp), producerTaskQueueType.value());
            // remove consumers
            barrierInfo.removeConsumers(barrierOp, consumersToRemove);
        }

        // try to optimize producers (2)
        const auto& barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
        auto consumerTaskQueueType = barrierInfo.haveSameImplicitDependencyTaskQueueType(barrierConsumers);
        if (consumerTaskQueueType.has_value() || barrierConsumers.empty()) {
            // barrier consumed by tasks with same type
            BarrierInfo::TaskSet producersToRemove;
            // find producers to remove
            if (barrierConsumers.empty()) {
                producersToRemove = barrierProducers;
            } else {
                producersToRemove = findExplicitDependencies(barrierProducers, consumerTaskQueueType.value());
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
    const auto barrierNum = barrierInfo.getNumOfVirtualBarriers();

    // Order barriers based on largest producer
    //
    // After already applied optimizations in this pass barrier state could have changed
    // and barriers might not have been ordered based on largest producer value (which corresponds to
    // largest barrier release time).
    // For compile time improvement - early termination of merge barrier logic, we need
    // barriers to be reordered so new vector is prepared that will be used as a base for iterating
    // over all barriers
    SmallVector<std::pair<size_t, std::optional<size_t>>> barIndAndMaxProdVec;
    barIndAndMaxProdVec.reserve(barrierNum);

    // Store number of barriers which do not have producers which nevertheless are not a candidate
    // for merge barriers logic. Later this value will be used to skip all the barriers
    // with no producers. After sorting barIndAndMaxProdVec they will be placed at the beginning
    size_t numOfBarriersWithNoProducers = 0;

    // For each barrier get the largest producer index
    for (size_t barrierInd = 0; barrierInd < barrierNum; ++barrierInd) {
        const auto producers = barrierInfo.getBarrierProducers(barrierInd);
        std::optional<size_t> maxProducer;
        if (producers.empty()) {
            numOfBarriersWithNoProducers++;
        } else {
            maxProducer = *std::max_element(producers.begin(), producers.end());
        }

        barIndAndMaxProdVec.push_back(std::make_pair(barrierInd, maxProducer));
    }

    // Sort the barrier indexes based on largest producer value. If barrier has no producers they will
    // be placed at the beginning
    llvm::sort(barIndAndMaxProdVec.begin(), barIndAndMaxProdVec.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second == rhs.second) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });

    const auto allProducersAfterConsumers = [](const BarrierInfo::TaskSet& producers,
                                               const BarrierInfo::TaskSet& consumers) {
        const auto maxConsumer = *std::max_element(consumers.begin(), consumers.end());
        const auto minProducer = *std::min_element(producers.begin(), producers.end());

        return minProducer > maxConsumer;
    };

    // Merge barriers if possible.
    // Skip initial barriers with no producers as they are not candidates for merge
    for (size_t ind = numOfBarriersWithNoProducers; ind < barrierNum; ++ind) {
        const auto barrierInd = barIndAndMaxProdVec[ind].first;
        auto barrierProducersA = barrierInfo.getBarrierProducers(barrierInd);
        if (barrierProducersA.empty()) {
            continue;
        }
        auto barrierConsumersA = barrierInfo.getBarrierConsumers(barrierInd);
        if (barrierConsumersA.empty()) {
            continue;
        }

        for (auto nextInd = ind + 1; nextInd < barrierNum; ++nextInd) {
            const auto nextBarrierInd = barIndAndMaxProdVec[nextInd].first;
            const auto barrierProducersB = barrierInfo.getBarrierProducers(nextBarrierInd);
            if (barrierProducersB.empty()) {
                continue;
            }
            const auto barrierConsumersB = barrierInfo.getBarrierConsumers(nextBarrierInd);
            if (barrierConsumersB.empty()) {
                continue;
            }

            // If for a given barrier B (nextBarrierInd) all producers are after all consumers of
            // barrier A (barrierInd) then neither this nor any later barrier will be a candidate to merge
            // with barrier A as they do not overlap their lifetime in schedule. Such early return is possible
            // because barriers are processed in order following barrier release time (latest producer)
            if (allProducersAfterConsumers(barrierProducersB, barrierConsumersA)) {
                break;
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
    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo);
    // Build the control relationship between any two task op. Note that the relationship includes the dependency by
    // the barriers as well as the implicit dependence by FIFO
    barrierInfo.buildTaskControlMap();

    // get original wait barrier map
    const auto origWaitBarriersMap = barrierInfo.getWaitBarriersMap();

    // DMA operation in the same FIFO do not require a barrier between them
    // optimize dependencies between DMA tasks in the same FIFO
    removeRedundantDependencies(func, barrierInfo);
    removeExplicitDependencies(func, barrierInfo);
    mergeBarriers(barrierInfo, origWaitBarriersMap);
    removeRedundantDependencies(func, barrierInfo);

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo);
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createDMABarrierOptimizationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMABarrierOptimizationPass(Logger log) {
    return std::make_unique<DMABarrierOptimizationPass>(log);
}
