//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

#include <llvm/ADT/SetOperations.h>

using namespace vpux;

namespace {

class SatisfyOneWaitBarrierPerTaskPass final :
        public VPURT::SatisfyOneWaitBarrierPerTaskBase<SatisfyOneWaitBarrierPerTaskPass> {
public:
    explicit SatisfyOneWaitBarrierPerTaskPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

BarrierInfo::TaskSet getBarriersProducerTasks(const BarrierInfo::TaskSet& waitBarriers, BarrierInfo& barrierInfo) {
    BarrierInfo::TaskSet newBarrierProducers;
    for (auto& waitBarrierInd : waitBarriers) {
        // merge all producers
        const auto barrierProducers = barrierInfo.getBarrierProducers(waitBarrierInd);
        llvm::set_union(newBarrierProducers, barrierProducers);
    }

    return newBarrierProducers;
}

bool canMergeBarriersForTasks(const BarrierInfo::TaskSet& producers, size_t availableSlots, BarrierInfo& barrierInfo) {
    size_t producerSlotCount = 0;
    for (auto& producerInd : producers) {
        producerSlotCount += barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(producerInd));

        if (producerSlotCount > availableSlots) {
            // exceeding producer slot count
            return false;
        }
    }

    return true;
}

void unlinkTaskFromParallelBarriers(size_t taskInd, const BarrierInfo::TaskSet& waitBarriers,
                                    BarrierInfo& barrierInfo) {
    // remove consumer task from parallel barriers
    for (auto& waitBarrierInd : waitBarriers) {
        const auto barrierConsumers = barrierInfo.getBarrierConsumers(waitBarrierInd);
        const auto waitBarrier = barrierInfo.getBarrierOpAtIndex(waitBarrierInd);

        // remove link from parallel barriers
        if (barrierConsumers.size() == 1) {
            // if only consumer, barrier can be reset
            barrierInfo.resetBarrier(waitBarrier);
        } else {
            barrierInfo.removeConsumer(taskInd, waitBarrier);
        }
    }
}

// merge wait barriers if task wait barrier count > 1 by
// creating a new barrier replacing parallel wait barriers
/*
        x..xn    y..ym             x..xn    y..ym
          |        |              /    \   /    \
        Bar0     Bar1     -->   Bar0    Bar1    Bar2
       /    \   /    \           |       |       |
    u..ui   o..oj   a..ak      u..ui   o..oj   a..ak
*/
void mergeLegalParallelProducers(size_t taskInd, VPURT::TaskOp taskOp, const BarrierInfo::TaskSet& parallelProducers,
                                 BarrierInfo& barrierInfo) {
    // create a new barrier for all parallel producers
    mlir::OpBuilder builder(taskOp);
    auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp.getLoc());
    barrierInfo.addNewBarrier(newBarrier);

    // add all legal producers to new barrier
    for (auto& barrierProducer : parallelProducers) {
        barrierInfo.addProducer(newBarrier, barrierProducer);
    }

    // add only current consumer
    barrierInfo.addConsumer(newBarrier, taskInd);
}

// try to create batches of producers using existing barriers if producers satisfy order constraint
SmallVector<BarrierInfo::TaskSet> createProducerBatches(const BarrierInfo::TaskSet& waitBarriers, size_t availableSlots,
                                                        BarrierInfo& barrierInfo) {
    SmallVector<BarrierInfo::TaskSet> legalBatches(1);

    auto prevLastUserInd = std::numeric_limits<size_t>::min();
    for (auto& waitBarrierInd : waitBarriers) {
        const auto barrierProducers = barrierInfo.getBarrierProducers(waitBarrierInd);

        auto firstUserInd = *std::min_element(barrierProducers.begin(), barrierProducers.end());
        if (firstUserInd > 0 && prevLastUserInd >= firstUserInd) {
            // producers do not satisfy order constraint
            return {};
        }

        prevLastUserInd = *std::max_element(barrierProducers.begin(), barrierProducers.end());

        auto currentBatchPlusBarrierProducers = legalBatches.back();
        llvm::set_union(currentBatchPlusBarrierProducers, barrierProducers);

        if (canMergeBarriersForTasks(currentBatchPlusBarrierProducers, availableSlots, barrierInfo)) {
            // can add to the same batch
            legalBatches.back() = std::move(currentBatchPlusBarrierProducers);
        } else {
            // need to create a new batch
            legalBatches.push_back(barrierProducers);
        }
    }

    return legalBatches;
}

// linearize wait barriers if task wait barrier count > 1 by
// linearizing legal batches of parallel wait barriers
/*
        x..xn    y..ym             x..xn
          |        |              /     \
        Bar0     Bar1     -->   Bar0     Bar1
       /    \   /    \           |        |
    u..ui   o..oj   a..ak      u..ui    y..ym
                                       /     \
                                     Bar2   Bar3
                                      |      |
                                    o..oj   a..ak
*/
void linearizeLegalParallelProducers(size_t taskInd, VPURT::TaskOp taskOp, const BarrierInfo::TaskSet& waitBarriers,
                                     const BarrierInfo::TaskSet& parallelProducers, size_t availableSlots,
                                     BarrierInfo& barrierInfo) {
    // create legal batches of barrier producers
    auto legalBatches = createProducerBatches(waitBarriers, availableSlots, barrierInfo);
    if (legalBatches.empty()) {
        // create new batches of producers that satisfy order constraint
        legalBatches = barrierInfo.createLegalVariantBatches(parallelProducers, availableSlots);
    }

    // last batch is the current task
    BarrierInfo::TaskSet nextBatch;
    nextBatch.insert(taskInd);

    mlir::OpBuilder builder(taskOp);

    // create a new barrier between batches
    for (const auto& batch : legalBatches | reversed) {
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp.getLoc());
        barrierInfo.addNewBarrier(newBarrier);

        for (auto& barrierProducer : batch) {
            barrierInfo.addProducer(newBarrier, barrierProducer);
        }

        for (auto& barrierConsumer : nextBatch) {
            barrierInfo.addConsumer(newBarrier, barrierConsumer);
        }

        nextBatch = batch;
    }
}

// modify barriers for task such that it is driven by a single wait barrier by merging parallel wait barriers
// or linearizing the producers of parallel wait barriers
bool ensureTaskDrivenBySingleBarrier(size_t taskInd, VPURT::TaskOp taskOp, const BarrierInfo::TaskSet& waitBarriers,
                                     size_t availableSlots, BarrierInfo& barrierInfo, Logger log) {
    log.trace("Got '{0}' parallel wait barriers for '{1}'", waitBarriers.size(), taskInd);

    auto parallelProducers = getBarriersProducerTasks(waitBarriers, barrierInfo);
    if (canMergeBarriersForTasks(parallelProducers, availableSlots, barrierInfo)) {
        log.nest().trace("Can merge '{0}' parallel producers for '{1}'", parallelProducers.size(), taskInd);
        mergeLegalParallelProducers(taskInd, taskOp, parallelProducers, barrierInfo);
    } else {
        log.nest().trace("Have to linearize '{0}' parallel producers for '{1}'", parallelProducers.size(), taskInd);
        linearizeLegalParallelProducers(taskInd, taskOp, waitBarriers, parallelProducers, availableSlots, barrierInfo);
    }

    log.trace("Unlink parallel barriers from '{0}'", taskInd);
    unlinkTaskFromParallelBarriers(taskInd, waitBarriers, barrierInfo);

    log.trace("Task '{0}', now has one parallel wait barrier", taskInd);
    return true;
}

void SatisfyOneWaitBarrierPerTaskPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();

    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);
    _log.trace("There are {0} slots for each barrier", maxAvailableSlots);

    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    auto availableSlots = maxAvailableSlots / 2;
    bool modifiedIR = false;

    // merge parallel wait barriers
    func->walk([&](VPURT::TaskOp taskOp) {
        const auto taskInd = barrierInfo.getIndex(taskOp);
        const auto waitBarriers = barrierInfo.getWaitBarriers(taskInd);
        if (waitBarriers.size() < 2) {
            // valid configuration
            return;
        }

        modifiedIR |= ensureTaskDrivenBySingleBarrier(taskInd, taskOp, waitBarriers, availableSlots, barrierInfo,
                                                      _log.nest());
    });

    if (!modifiedIR) {
        // IR was not modified
        barrierInfo.clearAttributes();
        return;
    }

    barrierInfo.orderBarriers();
    barrierInfo.updateIR();
    barrierInfo.clearAttributes();
    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createSatisfyOneWaitBarrierPerTaskPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSatisfyOneWaitBarrierPerTaskPass(Logger log) {
    return std::make_unique<SatisfyOneWaitBarrierPerTaskPass>(log);
}
