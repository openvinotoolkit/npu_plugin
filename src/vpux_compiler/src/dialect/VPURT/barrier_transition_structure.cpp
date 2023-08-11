//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"

using namespace vpux::VPURT;

BarrierScheduler::barrierTransitionStructure::barrierTransitionStructure(BarrierScheduler& feasibleBarrierScheduler,
                                                                         size_t taskCount, size_t time)
        : _feasibleBarrierScheduler(feasibleBarrierScheduler), _taskCount(taskCount), _time(time), _producers() {
}

void BarrierScheduler::barrierTransitionStructure::init() {
    _time = std::numeric_limits<size_t>::max();
    _previousBarrierTask = NULL;
    _currentBarrierTask = NULL;
    _producers.clear();
}

bool BarrierScheduler::barrierTransitionStructure::processNextScheduledTask(const ScheduledOpInfo& sinfo,
                                                                            mlir::OpBuilder& builder) {
    size_t currentTime = sinfo._scheduleTime;
    bool createdNewBarrierTask = false;

    _feasibleBarrierScheduler._log.trace("The scheduled time is {0}, the global time is {1}, the task is {2} the "
                                         "barrier index is {3} and the slot cout is {4}",
                                         sinfo._scheduleTime, _time, BarrierScheduler::getUniqueID(sinfo._op),
                                         sinfo._barrierIndex, sinfo._producerSlotCount);
    _feasibleBarrierScheduler._log = _feasibleBarrierScheduler._log.nest();
    if (_time != currentTime) {
        _feasibleBarrierScheduler._log.trace("Case 1: temporal transition happened, create a new barrier task");

        // Case-1: a temporal transition happened
        createdNewBarrierTask = true;
        maintainInvariantTemporalChange(sinfo, builder);
        _time = currentTime;
    } else {
        // Case-2: a trivial case
        _feasibleBarrierScheduler._log.trace("Case-2: trivial case - adding the scheduled task to the producer list");
        addScheduledTaskToProducerList(sinfo);
    }
    _feasibleBarrierScheduler._log = _feasibleBarrierScheduler._log.unnest();
    return createdNewBarrierTask;
}

void BarrierScheduler::barrierTransitionStructure::closeBarrierProducerList() {
    if (_currentBarrierTask == NULL) {
        return;
    }
    processCurrentBarrierProducerListCloseEvent(_currentBarrierTask, _previousBarrierTask);
}

inline void BarrierScheduler::barrierTransitionStructure::processCurrentBarrierProducerListCloseEvent(
        mlir::Operation* currentBarrier, mlir::Operation* previousBarrier) {
    _feasibleBarrierScheduler._log.trace("Process current barrier producer list close event");

    mlir::Operation* barrierEnd = NULL;
    VPUX_THROW_UNLESS(currentBarrier != barrierEnd, "Error the current barrier is Null");

    // Get the barrier object for the three barrier tasks
    mlir::Operation* barrierPrevious = NULL;

    if (previousBarrier != barrierEnd) {
        barrierPrevious = previousBarrier;
    }

    _feasibleBarrierScheduler._log.trace("The ID of barrier b_curr is {0}", currentBarrier->getAttr(virtualIdAttrName));
    size_t currentBarrierID = getBarrierUniqueID(currentBarrier);
    _feasibleBarrierScheduler._log = _feasibleBarrierScheduler._log.nest();

    for (producerIteratorType producer = _producers.begin(); producer != _producers.end(); ++producer) {
        mlir::Operation* source = *producer;
        size_t sourceID = getTaskUniqueID(source);

        // Step-1.2 (a): producers

        if (currentBarrierID < _feasibleBarrierScheduler._configureBarrierOpWaitMap.size()) {
            _feasibleBarrierScheduler._log.trace("Adding producer task with ID {0} to barrier ID {1}",
                                                 BarrierScheduler::getUniqueID(source), currentBarrierID);

            _feasibleBarrierScheduler._configureBarrierOpWaitMap[currentBarrierID].set((unsigned)sourceID);
        } else {
            VPUX_THROW("Error unable to find the update tasks for barrier ID {0}",
                       currentBarrier->getAttr(virtualIdAttrName));
        }

        // Step-1.2 (b): consumers

        if (currentBarrierID < _feasibleBarrierScheduler._configureBarrierOpUpdateMap.size()) {
            auto opConsumers = _feasibleBarrierScheduler.getConsumerOps(source);

            for (auto consumer = opConsumers.begin(); consumer != opConsumers.end(); ++consumer) {
                _feasibleBarrierScheduler._log.trace("Step-1.2 Adding consumer task ID {0} to barrier ID {1}",
                                                     BarrierScheduler::getUniqueID(*consumer),
                                                     currentBarrier->getAttr(virtualIdAttrName));

                size_t consumerID = getTaskUniqueID(*consumer);
                _feasibleBarrierScheduler._configureBarrierOpUpdateMap[currentBarrierID].set((unsigned)consumerID);
            }
        } else {
            VPUX_THROW("Error unable to find the wait tasks for barrier ID {0}",
                       currentBarrier->getAttr(virtualIdAttrName));
        }

        // Step-1.3
        if (barrierPrevious) {
            size_t barrierPreviousID = getBarrierUniqueID(barrierPrevious);

            if (barrierPreviousID < _feasibleBarrierScheduler._configureBarrierOpUpdateMap.size()) {
                _feasibleBarrierScheduler._log.trace("Step-1.3 Adding consumer task ID {0} to barrier ID {1}",
                                                     BarrierScheduler::getUniqueID(source),
                                                     barrierPrevious->getAttr(virtualIdAttrName));

                _feasibleBarrierScheduler._configureBarrierOpUpdateMap[barrierPreviousID].set((unsigned)sourceID);
            } else {
                VPUX_THROW("Not found");
            }
        }
    }  // foreach producer
    _feasibleBarrierScheduler._log = _feasibleBarrierScheduler._log.unnest();
}

void BarrierScheduler::barrierTransitionStructure::maintainInvariantTemporalChange(const ScheduledOpInfo& sinfo,
                                                                                   mlir::OpBuilder& builder) {
    _feasibleBarrierScheduler._log = _feasibleBarrierScheduler._log.nest();
    _feasibleBarrierScheduler._log.trace("Maintain an invariant temporal change, the scheduled time is {0}, task ID is "
                                         "{1} the barrier index is {2} and the slot cout is {3}",
                                         sinfo._scheduleTime, BarrierScheduler::getUniqueID(sinfo._op),
                                         sinfo._barrierIndex, sinfo._producerSlotCount);

    //              B_prev
    // curr_state : Prod_list={p_0, p_1, ... p_n}-->B_curr
    // event: Prod_list={q_0}->B_curr_new
    //
    // scheduler says it want to associate both B_old and B_new to the
    // same physical barrier.
    //
    // Restore Invariant:
    // Step-1.1: create a new barrier task (B_new).
    // Step-1.2: update B_curr
    //        a. producers: B_curr is now closed so update its producers
    //        b. consumers: for each (p_i, u) \in P_old x V
    //                      add u to the consumer list of B_old
    // Step-1.3: update B_prev
    //           consumers: add p_i \in P_old to the consumer list of
    //                      B_prev. This is because B_prev and B_curr
    //                      are associated with same physical barrier.
    // Step-2: B_prev = B_curr , B_curr = B_curr_new , Prod_list ={q0}
    mlir::Operation* previousBarrier = _previousBarrierTask;
    mlir::Operation* currentBarrier = _currentBarrierTask;
    mlir::Operation* barrierEnd = NULL;
    mlir::Operation* newCurrentBarrier = barrierEnd;

    newCurrentBarrier = createNewBarrierTask(sinfo, builder);
    VPUX_THROW_UNLESS(newCurrentBarrier != barrierEnd, "Error newly created barrier is Null");

    // STEP-1
    if (currentBarrier != barrierEnd) {
        _feasibleBarrierScheduler._log.trace("The ID of barrier currentBarrier is {0}",
                                             currentBarrier->getAttr(virtualIdAttrName));
        processCurrentBarrierProducerListCloseEvent(currentBarrier, previousBarrier);
    }

    // STEP-2
    _previousBarrierTask = _currentBarrierTask;
    _currentBarrierTask = newCurrentBarrier;
    _producers.clear();
    addScheduledTaskToProducerList(sinfo);
    _feasibleBarrierScheduler._log = _feasibleBarrierScheduler._log.unnest();
}

void BarrierScheduler::barrierTransitionStructure::addScheduledTaskToProducerList(const ScheduledOpInfo& sinfo) {
    auto scheduled_op = sinfo._op;

    _feasibleBarrierScheduler._log.trace("Adding task {0} to the producer list",
                                         BarrierScheduler::getUniqueID(sinfo._op));
    _producers.insert(scheduled_op);
}

mlir::Operation* BarrierScheduler::barrierTransitionStructure::createNewBarrierTask(const ScheduledOpInfo& sinfo,
                                                                                    mlir::OpBuilder& builder) {
    _feasibleBarrierScheduler._log.trace("Creating a new virtual barrier task");

    size_t barrierTaskId = _feasibleBarrierScheduler._configureBarrierOpWaitMap.size();

    auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(sinfo._op->getLoc());
    newBarrier->setAttr(virtualIdAttrName, getIntAttr(newBarrier->getContext(), barrierTaskId));
    _feasibleBarrierScheduler._orderedBarrier.push_back(newBarrier);

    llvm::BitVector newBarrierProducers;
    newBarrierProducers.resize((unsigned)_taskCount);
    llvm::BitVector newBarrierConsumers;
    newBarrierConsumers.resize((unsigned)_taskCount);

    _feasibleBarrierScheduler._configureBarrierOpWaitMap.push_back(newBarrierProducers);
    _feasibleBarrierScheduler._configureBarrierOpUpdateMap.push_back(newBarrierConsumers);

    _feasibleBarrierScheduler._log.trace("Created a new barrier task with barrier ID {0} after task id {1}",
                                         barrierTaskId, BarrierScheduler::getUniqueID(sinfo._op));

    return newBarrier;
}

size_t BarrierScheduler::barrierTransitionStructure::getTaskUniqueID(mlir::Operation* task) {
    return checked_cast<size_t>(
            mlir::dyn_cast<VPURT::TaskOp>(task)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
}

size_t BarrierScheduler::barrierTransitionStructure::getBarrierUniqueID(mlir::Operation* task) {
    return checked_cast<size_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(task)
                                        ->getAttr(virtualIdAttrName)
                                        .cast<mlir::IntegerAttr>()
                                        .getInt());
}
