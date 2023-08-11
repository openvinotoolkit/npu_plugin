//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/utils/dma.hpp"

using namespace vpux;

namespace {

// check if taskA and taskB use the same barriers
bool useSameBarriers(size_t taskA, size_t taskB, BarrierInfo& barrierInfo) {
    if (barrierInfo.getWaitBarriers(taskA) != barrierInfo.getWaitBarriers(taskB)) {
        return false;
    }

    return barrierInfo.getUpdateBarriers(taskA) == barrierInfo.getUpdateBarriers(taskB);
}

// linearize tasks to execute sequentially
/*
                                   TaskOp1
                                      |
                                     Bar1
                                      |
    TaskOp1...TaskOpM     =>         ...
                                      |
                                     BarM
                                      |
                                   TaskOpM
*/
bool linearizeTasks(std::set<size_t>& linearizationTasks, BarrierInfo& barrierInfo, Logger log) {
    log.trace("Linearizing tasks");

    if (linearizationTasks.size() < 2) {
        log.trace("Can not linearize '{0}' tasks", linearizationTasks.size());
        return false;
    }

    // account for parallel tasks with same wait & update barrier(s) which do not produce new barriers
    const auto findEndItr = [&](std::set<size_t>::iterator currItr) {
        auto endItr = currItr;
        ++endItr;
        while (endItr != linearizationTasks.end() && useSameBarriers(*currItr, *endItr, barrierInfo)) {
            ++endItr;
        }
        return endItr;
    };

    auto currTask = linearizationTasks.begin();
    auto nextTask = currTask;

    auto earliestConsumer = barrierInfo.getTaskOpAtIndex(*currTask);
    mlir::OpBuilder builder(earliestConsumer);
    builder.setInsertionPoint(earliestConsumer);

    // linearize all tasks
    bool linearized = false;
    while (currTask != linearizationTasks.end() && nextTask != linearizationTasks.end()) {
        // find sections of tasks using same barriers
        auto producersEnd = findEndItr(currTask);
        nextTask = producersEnd;
        if (nextTask == linearizationTasks.end()) {
            break;
        }
        auto consumersEnd = findEndItr(nextTask);

        // skip if barrier already exists
        // TODO: E#80600 also check FIFO dependency
        if (barrierInfo.controlPathExistsBetween(*currTask, *nextTask)) {
            currTask = nextTask;
            ++nextTask;
            continue;
        }

        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(earliestConsumer->getLoc());
        log.trace("Created new barrier '{0}'", newBarrier);
        barrierInfo.addNewBarrier(newBarrier);

        while (currTask != producersEnd) {
            log.nest().trace("Add producer to new barrier '{0}'", *currTask);
            barrierInfo.addProducer(newBarrier, *currTask);
            ++currTask;
        }

        while (nextTask != consumersEnd) {
            log.nest().trace("Add consumer to new barrier '{0}'", *nextTask);
            barrierInfo.addConsumer(newBarrier, *nextTask);
            ++nextTask;
        }

        linearized = true;
    }

    return linearized;
}

// linearize barriers such that tasks execute sequentially
/*
    TaskOp1...TaskOpK       TaskOp1 - Bar - ... - Bar - TaskOpK
            |                                |
       Bar1...BarN      =>              Bar1...BarN
            |                                |
    TaskOp1...TaskOpM       TaskOp1 - Bar - ... - Bar - TaskOpm
*/
void linearizeBarriers(mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>& barrierOps, BarrierInfo& barrierInfo,
                       Logger log) {
    log.trace("Linearizing barriers");

    // store barrier producers and consumers to linearize
    std::set<size_t> tasksToLinearize;
    for (const auto& barrierOp : barrierOps) {
        log.nest().trace("Barrier '{0}'", barrierOp);

        const auto barrierProducers = barrierInfo.getBarrierProducers(barrierOp);
        tasksToLinearize.insert(barrierProducers.begin(), barrierProducers.end());

        const auto barrierConsumers = barrierInfo.getBarrierConsumers(barrierOp);
        tasksToLinearize.insert(barrierConsumers.begin(), barrierConsumers.end());
    }

    // TODO: try to efficiently linearize producers or consumers only
    // Note: might increase compilation time

    // linearize producers and consumers
    auto linearized = linearizeTasks(tasksToLinearize, barrierInfo, log);
    log.trace("Linearized = '{0}' producers and consumers", linearized);
    linearizeTasks(tasksToLinearize, barrierInfo, log);
    log.trace("Linearized producers and consumers");
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

class ReduceExceedingActiveCountBarriersPass final :
        public VPURT::ReduceExceedingActiveCountBarriersBase<ReduceExceedingActiveCountBarriersPass> {
public:
    explicit ReduceExceedingActiveCountBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ReduceExceedingActiveCountBarriersPass::safeRunOnFunc() {
    auto func = getOperation();

    const auto numBarriersToUse = numBarriers.hasValue() ? checked_cast<size_t>(numBarriers.getValue())
                                                         : checked_cast<size_t>(VPUIP::getNumAvailableBarriers(func));
    const auto maxAvailableSlots = maxVariantCount.hasValue() ? checked_cast<size_t>(maxVariantCount.getValue())
                                                              : VPUIP::getBarrierMaxVariantCount(func);

    _log.trace("There are {0} physical barriers and {1} slots for each barrier", numBarriersToUse, maxAvailableSlots);

    VPUX_THROW_UNLESS(numBarriersToUse > 1, "Not possible to satisfy barrier requirement numBarriersToUse '{0}'",
                      numBarriersToUse);

    auto& barrierInfo = getAnalysis<BarrierInfo>();
    if (barrierInfo.getNumOfVirtualBarriers() <= numBarriersToUse) {
        _log.trace("Fewer barriers '{0}', than physical barriers.", barrierInfo.getNumOfVirtualBarriers());
        barrierInfo.clearAttributes();
        return;
    }

    VPURT::BarrierSimulator barrierSim(func);
    VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");

    auto barSimLog = _log.nest();
    barSimLog.setName("barrier-schedule-sim");
    if (mlir::succeeded(barrierSim.simulateBarriers(barSimLog, numBarriersToUse))) {
        _log.trace("Barrier simulation passed with '{0}', no isses with exceeding barriers", numBarriersToUse);
        barrierInfo.clearAttributes();
        return;
    }

    SmallVector<mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>> barrierBatchesToLegalize;

    const auto updateAnalysis = [&]() {
        barrierInfo.optimizeBarriers();
        barrierInfo.buildTaskControllMap(false);
        // TODO: E#80635 satisfy SRP, do not modify IR
        barrierInfo.orderBarriers();
        barrierInfo.updateIR();

        barrierSim = VPURT::BarrierSimulator{func};
        if (mlir::succeeded(barrierSim.simulateBarriers(barSimLog, numBarriersToUse, true))) {
            _log.trace("Barrier simulation passed with '{0}', no isses with exceeding barriers", numBarriersToUse);
            VPUX_THROW_UNLESS(barrierBatchesToLegalize.empty(),
                              "Simulation passed, but '{0}' batches to legalize exist",
                              barrierBatchesToLegalize.size());
        }
        barrierBatchesToLegalize = barrierSim.getBarrierBatchesToLegalize();
    };

    // optimize current barrier state and perform simulation
    updateAnalysis();

    // iterate through barrier batches to legalize and reduce active barrier count in each batch
    for (size_t it = 0; it < barrierInfo.getNumOfVirtualBarriers() && !barrierBatchesToLegalize.empty(); ++it) {
        _log.trace("Iteration '{0}', there are '{1}' batches", it, barrierBatchesToLegalize.size());

        for (auto& activeBarriers : barrierBatchesToLegalize) {
            _log.trace("There are '{0}' active barriers, reduce active barrier count", activeBarriers.size());

            VPUX_THROW_UNLESS(activeBarriers.size() > 0,
                              "Failed to retrieve active barriers from barrier simulation, got '{0}' active barriers",
                              activeBarriers.size());

            // TODO: E#71194 try to merge active barriers
            // TODO: E#71585 use task cycle info
            // Note: currently not merging barriers due to worse performance

            // linearize execution
            linearizeBarriers(activeBarriers, barrierInfo, _log);
        }

        // TODO: merge more barriers and split exceeding active count barriers ?

        updateAnalysis();
    }

    // remove attributes before removing barriers
    barrierInfo.clearAttributes();
    postProcessBarrierOps(func);
}

}  // namespace

//
// createReduceExceedingActiveCountBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createReduceExceedingActiveCountBarriersPass(Logger log) {
    return std::make_unique<ReduceExceedingActiveCountBarriersPass>(log);
}
