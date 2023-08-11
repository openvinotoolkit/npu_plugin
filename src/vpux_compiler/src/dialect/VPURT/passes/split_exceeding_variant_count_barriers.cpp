//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

using namespace vpux;

namespace {

class SplitExceedingVariantCountBarriersPass final :
        public VPURT::SplitExceedingVariantCountBarriersBase<SplitExceedingVariantCountBarriersPass> {
public:
    explicit SplitExceedingVariantCountBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// create batches of tasks such that for every batch the total used barrier slot count is legal
SmallVector<mlir::DenseSet<size_t>> createLegalVariantBatches(const BarrierInfo::TaskSet& tasks, size_t availableSlots,
                                                              BarrierInfo& barrierInfo) {
    // store batches of tasks
    SmallVector<mlir::DenseSet<size_t>> legalBatches;
    legalBatches.push_back({});
    // store total slot count used by batch
    size_t totalSlotCount = 0;

    const auto isLegalVariantCountWith = [&](size_t numSlotsUsedByTask) -> bool {
        return (totalSlotCount + numSlotsUsedByTask) <= availableSlots;
    };

    // create batches for new barriers
    for (const auto& linearizationTask : tasks) {
        // find number of slots consumed by this task
        auto numSlotsUsedByTask = barrierInfo.getNumOfSlotsUsed(barrierInfo.getTaskOpAtIndex(linearizationTask));

        // check if new batch needs to be created
        if (!isLegalVariantCountWith(numSlotsUsedByTask)) {
            legalBatches.push_back({});
            totalSlotCount = 0;
        }

        legalBatches.rbegin()->insert(linearizationTask);
        totalSlotCount += numSlotsUsedByTask;
    }

    return legalBatches;
}

// split barriers if producer count > AVAILIBLE SLOTS
// will produce ceil(NUM PRODUCERS / AVAILIBLE SLOTS) barriers
/*
    x1  x2  ... xn             x1  ... x256   x257 ... xn
     \  \   /   /               \   |   /        \  |  /
         Bar0         -->          Bar0            Bar1
          |                            \          /
          u0                                u0
*/
void splitBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots, BarrierInfo& barrierInfo,
                           Logger log) {
    // producers will be divided into batches
    auto barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
    auto barrierProducersCount = barrierProducers.size();
    log.trace("Got barrier: '{0}' at '{1}' with barrier producer count '{2}'", barrierOp->getName(),
              barrierOp->getLoc(), barrierProducersCount);

    // consumers remain the same for all produced batched barriers
    const auto barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
    // crete batches for producers
    auto producerBatches = createLegalVariantBatches(barrierProducers, availableSlots, barrierInfo);

    mlir::OpBuilder builder(barrierOp);

    // create barriers based on batches
    for (const auto& producerBatch : producerBatches) {
        builder.setInsertionPoint(barrierOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(barrierOp->getLoc());
        log.trace("Created new barrier '{0}'", newBarrier);
        barrierInfo.addNewBarrier(newBarrier);

        for (const auto& producer : producerBatch) {
            log.nest().trace("Add producer to new barrier '{0}'", producer);
            barrierInfo.addProducer(newBarrier, producer);
        }

        for (const auto& consumer : barrierConsumers) {
            log.nest().trace("Add consumer to new barrier '{0}'", consumer);
            barrierInfo.addConsumer(newBarrier, consumer);
        }
    }

    barrierInfo.resetBarrier(barrierOp);
    log.trace("Barrier successfuly replaced with batch of barriers");
}

// split barriers if consumer count > AVAILIBLE SLOTS
// will produce ceil(NUM PRODUCERS / AVAILIBLE SLOTS) barriers
/*
          u0                                u0
          |                            /          \
         Bar0         -->          Bar0            Bar1
     /  /   \   \               /   |   \        /  |  \
    x1  x2  ... xn             x1  ... x256   x257 ... xn
*/
void splitBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, size_t availableSlots, BarrierInfo& barrierInfo,
                           Logger log) {
    // consumers will be divided into batches
    auto barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
    auto barrierConsumersCount = barrierConsumers.size();
    log.trace("Got barrier: '{0}' at '{1}' with barrier consumer count '{2}'", barrierOp->getName(),
              barrierOp->getLoc(), barrierConsumersCount);

    // producers remain the same for all produced batched barriers
    const auto barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
    // crete batches for consumers
    auto consumerBatches = createLegalVariantBatches(barrierConsumers, availableSlots, barrierInfo);

    mlir::OpBuilder builder(barrierOp);

    // create barriers based on batches
    for (const auto& consumerBatch : consumerBatches) {
        builder.setInsertionPoint(barrierOp);
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(barrierOp->getLoc());
        log.trace("Created new barrier '{0}'", newBarrier);
        barrierInfo.addNewBarrier(newBarrier);

        for (const auto& producer : barrierProducers) {
            log.nest().trace("Add producer to new barrier '{0}'", producer);
            barrierInfo.addProducer(newBarrier, producer);
        }

        for (const auto& consumer : consumerBatch) {
            log.nest().trace("Add consumer to new barrier '{0}'", consumer);
            barrierInfo.addConsumer(newBarrier, consumer);
        }
    }

    barrierInfo.resetBarrier(barrierOp);
    log.trace("Barrier successfuly replaced with batch of barriers");
}

void SplitExceedingVariantCountBarriersPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& barrierInfo = getAnalysis<BarrierInfo>();

    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);
    _log.trace("There are {0} slots for each barrier", maxAvailableSlots);

    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    auto availableSlots = maxAvailableSlots / 2;

    // verify each task individually satisifies variant count
    func->walk([&](VPURT::TaskOp taskOp) {
        VPUX_THROW_UNLESS(!mlir::isa<VPUIP::NCEClusterTilingOp>(taskOp.getInnerTaskOp()),
                          "Inner task op wrapped with NCEClusterTilingOp '{0}'", taskOp);
        VPUX_THROW_UNLESS(barrierInfo.getNumOfSlotsUsed(taskOp) <= availableSlots,
                          "Task '{0}' uses too many barrier slots '{1}'", barrierInfo.getNumOfSlotsUsed(taskOp),
                          taskOp);
    });

    const auto removeBarriersWithNoUse = [&]() {
        func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
            if (barrierOp.barrier().use_empty()) {
                barrierOp->erase();
            }
        });
    };

    // Note: profiling parser logic assumes that
    // invariants of the same layer should use the same barriers

    // split barriers if needed
    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        if (barrierInfo.getProducerSlotCount(barrierOp) <= availableSlots) {
            // valid producer configuration - barrier is legal
            return;
        }

        splitBarrierProducers(barrierOp, availableSlots, barrierInfo, _log);
    });

    // split barriers if needed
    func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        if (barrierInfo.getConsumerSlotCount(barrierOp) <= availableSlots) {
            // valid consumer configuration - barrier is legal
            return;
        }

        splitBarrierConsumers(barrierOp, availableSlots, barrierInfo, _log);
    });

    barrierInfo.updateIR();
    barrierInfo.clearAttributes();
    removeBarriersWithNoUse();
}

}  // namespace

//
// createSplitExceedingVariantCountBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSplitExceedingVariantCountBarriersPass(Logger log) {
    return std::make_unique<SplitExceedingVariantCountBarriersPass>(log);
}
