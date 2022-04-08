//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/cycle_based_barrier_scheduler.hpp"

using namespace vpux::VPURT;

//
// Barrier Scheduler
//
// Barriers are physical synchronization primitives in VPU hardware. Every task
// requires a barrier resource to indicate its completion to its adjacent tasks.
// Thus every scheduled operation with at least one outgoing edge requires an update barrier.
//
// In the feasible memory scheduler, a feasible schedule is generated. However, VPU hardware
// only has a finite number of barriers, 8 per cluster. The Barrier scheduler class ensures
// that in all possible schedules of the number of active barriers does not exceed the available
// physical barriers for the target platform. To achieve this the scheduler may need to insert
// some additional control dependencies among the tasks.

// The barrier scheduler, similar to the feasible memory scheduler, is list scheduler with access to a
// global resource state. In the feasible memory scheduler, the resource state (the resource being managed)
// is memory. In the barrier scheduler, the resource being managed in the number of available barriers and
// the number of producers to a barrier.

// The hardware allows a finite producer count of 256 for each of the update barriers.
// This means that multiple tasks can update the same active barrier. This is incorporated into the
// barrier resource model by using the term "barrier slots".
// In addition to the upper bound of available barriers it is assumed that each of these barriers has
// a maximum of 256 slots. The barrier demand is expressed as the number of slots required.
// In the context of VPU hardware the number of slots for a DPU tasks are the DPU workloads
// and for a DMA/UPA tasks it is 1.

// The feasible memory scheduler (see FeasibleAllocationPass) generates a cycle start time and cycle end time for each
// task in a schedule. Cycle times refer to the clock cycles on the hardware that are used to perform an inference.
// For example, a task can start at cycle 5 and ends at cycle 10. The next task could start at cycle 11 or a task could
// be running in parallel during cycle 5 and 10.

// Barrier scheduler will generate a list of time stamp including all possible cycle values.
// At each time t,
// 1) if a task's cycle start <= t, then assign barrier for it and update related barrier consumer.
// 2) if a task's cycle end <= t, then update the related barrier producer.

CycleBasedBarrierScheduler::CycleBasedBarrierScheduler(mlir::FuncOp func, Logger log)
        : _barrierCount(),
          _slotsPerBarrier(),
          _barrierResourceState(),
          _heap(),
          _currentTime(0),
          _barrierResourceUtilizationMap(),
          _barrierMap(),
          _configureBarrierOpWaitTask(),
          _configureBarrierOpUpdateTask(),
          _configureTaskOpWaitBarrier(),
          _configureTaskOpUpdateBarrier(),
          _taskConsumerMapOriginal(),
          _log(log),
          _func(func),
          _enableDMAOptimization(false),
          _taskCount(){};

// Explicit dependency will be converted to barrier dependency during scheduling. So we want to reduce the dependency as
// much as posssible before scheduling. It helps to use barrier more efficiently.

// Actually the redundant explicit dependency is already optimized by OptimizeAsyncDeps pass before barrier scheduling.
// Also feasible_allocation pass will linearize all executors thus the only dependencies that barrier scheduler need to
// optimize is to remove unnecessary dependency between DMAs, which can execute in FIFO manner.

// But we still keep the part of optimizting redundant explicit dependency, because
// 1) it makes barrier scheduler doesn't rely on previous pass to optimize dependency
// 2) it takes little time as nothing to optimize

void CycleBasedBarrierScheduler::optimizeDependency() {
    _log.trace("Optimize dependency before scheduling");
    _log = _log.nest();
    _configureTaskOpWaitBarrier.resize(_taskCount);
    for (auto& wait : _configureTaskOpWaitBarrier) {
        wait.resize(static_cast<unsigned>(_taskCount));
    }

    _configureTaskOpUpdateBarrier.resize(_taskCount);
    for (auto& update : _configureTaskOpUpdateBarrier) {
        update.resize(static_cast<unsigned>(_taskCount));
    }

    // Assign individual barrier for each task to build dependency map
    std::set<VPURT::TaskOp> allTasks;
    for (auto taskList : _orderedTasksByCycleStart) {
        std::copy(taskList.second.begin(), taskList.second.end(), std::inserter(allTasks, allTasks.end()));
    }

    SmallVector<VPURT::TaskOp> allTasksSortedByCycleStart(allTasks.begin(), allTasks.end());
    std::sort(allTasksSortedByCycleStart.begin(), allTasksSortedByCycleStart.end(), startCycleTaskComparator());

    size_t schedulingNumber = 0UL;
    for (auto taskOp : allTasksSortedByCycleStart) {
        auto virtualBarrierID = schedulingNumber;
        auto taskId = getUniqueID(taskOp.getOperation());
        taskOp.getOperation()->setAttr(schedulingNumberAttrName, getIntAttr(taskOp.getContext(), schedulingNumber));

        llvm::BitVector newBarrierProducers;
        newBarrierProducers.resize(static_cast<unsigned>(_taskCount));
        llvm::BitVector newBarrierConsumers;
        newBarrierConsumers.resize(static_cast<unsigned>(_taskCount));

        newBarrierProducers.set(static_cast<unsigned>(taskId));
        _configureTaskOpUpdateBarrier[taskId].set(static_cast<unsigned>(virtualBarrierID));

        for (auto origConsumer : _taskConsumerMapOriginal[taskOp.getOperation()]) {
            auto consumerId = getUniqueID(origConsumer);
            newBarrierConsumers.set(static_cast<unsigned>(consumerId));
            _configureTaskOpWaitBarrier[consumerId].set(static_cast<unsigned>(virtualBarrierID));
        }

        _configureBarrierOpWaitTask.push_back(newBarrierProducers);
        _configureBarrierOpUpdateTask.push_back(newBarrierConsumers);

        schedulingNumber++;
    }

    _log.trace("Removing redundant dependencies");
    removeRedundantDependencies();

    _log.trace("Removing redundant barriers");
    removeRedundantBarriers(true);

    _log.trace("Removing redundant dependencies after barrier merge");
    removeRedundantDependencies();

    _log.trace("Update consumer map after dependency optimization");
    updateDependency();
    _log = _log.unnest();
}

// Update consumer map after dependency optimization
void CycleBasedBarrierScheduler::updateDependency() {
    _taskConsumerMapOriginal.clear();
    for (size_t ind = 0; ind < _configureBarrierOpWaitTask.size(); ind++) {
        auto& producers = _configureBarrierOpWaitTask[ind];
        auto& consumers = _configureBarrierOpUpdateTask[ind];
        std::set<mlir::Operation*> consumersList;
        size_t firstStart = std::numeric_limits<size_t>::max();
        for (auto cons = consumers.set_bits_begin(); cons != consumers.set_bits_end(); cons++) {
            consumersList.insert(_orderedTasks[*cons].getOperation());
            if (_operationStartCycle[*cons] < firstStart) {
                firstStart = _operationStartCycle[*cons];
            }
        }

        size_t lastEnd = 0;
        for (auto prod = producers.set_bits_begin(); prod != producers.set_bits_end(); prod++) {
            for (auto cons : consumersList) {
                _taskConsumerMapOriginal[_orderedTasks[*prod].getOperation()].insert(cons);
                _log.trace("Add control edge {0} -> {1}", *prod, getUniqueID(cons));
            }
            if (_operationEndCycle[*prod] > lastEnd) {
                lastEnd = _operationEndCycle[*prod];
            }
        }

        if (lastEnd > firstStart) {
            _log.trace("Invalid dependency exists : last producer ends at {0} while first consumer starts at {1}",
                       lastEnd, firstStart);
        }
    }

    _configureBarrierOpWaitTask.clear();
    _configureBarrierOpUpdateTask.clear();
    _configureTaskOpWaitBarrier.clear();
    _configureTaskOpUpdateBarrier.clear();
    _pathLookUpTable.clear();
}

void CycleBasedBarrierScheduler::optimizeIRDependency() {
    std::map<mlir::Operation*, std::pair<SmallVector<mlir::Operation*>, SmallVector<mlir::Operation*>>>
            barrierOpUpdateWaitMap;
    const auto updateBarrierConfigs = [&](VPURT::TaskOp taskOp) {
        for (const auto bar : taskOp.waitBarriers()) {
            auto iter = barrierOpUpdateWaitMap.find(bar.getDefiningOp());
            if (iter != barrierOpUpdateWaitMap.end()) {
                barrierOpUpdateWaitMap[bar.getDefiningOp()].second.push_back(taskOp);
            } else {
                SmallVector<mlir::Operation*> producers{};
                SmallVector<mlir::Operation*> consumers{taskOp};
                barrierOpUpdateWaitMap.insert(
                        std::make_pair(bar.getDefiningOp(), std::make_pair(producers, consumers)));
            }
        }

        for (const auto bar : taskOp.updateBarriers()) {
            auto iter = barrierOpUpdateWaitMap.find(bar.getDefiningOp());
            if (iter != barrierOpUpdateWaitMap.end()) {
                barrierOpUpdateWaitMap[bar.getDefiningOp()].first.push_back(taskOp);
            } else {
                SmallVector<mlir::Operation*> producers{taskOp};
                SmallVector<mlir::Operation*> consumers{};
                barrierOpUpdateWaitMap.insert(
                        std::make_pair(bar.getDefiningOp(), std::make_pair(producers, consumers)));
            }
        }
    };

    // Compute original consumers of tasks
    const auto configOriginalConsumers = [&](VPURT::TaskOp taskOp) {
        std::set<mlir::Operation*> consumers;
        for (const auto bar : taskOp.updateBarriers()) {
            auto iter = barrierOpUpdateWaitMap.find(bar.getDefiningOp());
            if (iter != barrierOpUpdateWaitMap.end()) {
                for (auto& barrierUpdateTask : iter->second.second) {
                    consumers.insert(barrierUpdateTask);
                }
            } else {
                VPUX_THROW("barrier '{0}' not found", bar.getDefiningOp());
            }
        }
        _taskConsumerMapOriginal.insert(std::make_pair(taskOp.getOperation(), consumers));
    };

    _func->walk([&](VPURT::TaskOp taskOp) {
        updateBarrierConfigs(taskOp);
    });

    _func->walk([&](VPURT::TaskOp taskOp) {
        configOriginalConsumers(taskOp);
    });

    optimizeDependency();

    // Remove the original virtual barriers, optimal barriers are inserted based on the generated schedule
    removeVirtualBarriers();

    // Remove temporary attribute
    _func->walk([](VPURT::TaskOp op) {
        op->removeAttr(schedulingNumberAttrName);
    });

    _log.trace("Removed all the original declare virtual barrier ops");
}

void CycleBasedBarrierScheduler::pushToScheduleTimeHeap(const HeapElement& elem) {
    _heap.push_back(elem);
    std::push_heap(_heap.begin(), _heap.end(), MinHeapOrdering());
}

llvm::Optional<CycleBasedBarrierScheduler::taskAndCyclePair>
CycleBasedBarrierScheduler::updateCycleStartTimeDueToNativeExecutorDependency(
        vpux::VPURT::TaskOp task, size_t newStartCycle, SmallVector<vpux::VPURT::TaskOp>& orderedTasksByCycleStart) {
    auto itr = std::find(orderedTasksByCycleStart.begin(), orderedTasksByCycleStart.end(), task);
    VPUX_THROW_UNLESS(itr != orderedTasksByCycleStart.end(), "task {0} is not found in list", task);
    itr++;
    if (itr != orderedTasksByCycleStart.end()) {
        auto nextTaskID = getUniqueID((*itr).getOperation());
        auto nextTaskStart = _operationStartCycle[nextTaskID];
        if (nextTaskStart < newStartCycle) {
            _log.trace("Update start cycle from {0} to {1} for consumer task {2}", nextTaskStart, newStartCycle,
                       nextTaskID);
            return std::make_pair(_orderedTasks[nextTaskID], newStartCycle);
        }
    }

    return None;
}

/// @brief update the start cycle of a task
/// @details when a task can't be scheduled in time, we need to update its start cycle. The new start cycle must be
/// larger than the orginial one. Meanwhile we need to check and update the start cycle of tasks which are waiting on
/// current task.
/// @param taskOp the task which needs update of start cycle
/// @param newStartCycle the new start cycle
void CycleBasedBarrierScheduler::updateCycleStartTime(vpux::VPURT::TaskOp srcTaskOp, size_t srcNewStartCycle) {
    SmallVector<std::pair<vpux::VPURT::TaskOp, size_t>> opsNeedCycleUpdate;
    opsNeedCycleUpdate.push_back(std::make_pair(srcTaskOp, srcNewStartCycle));

    while (!opsNeedCycleUpdate.empty()) {
        auto opNeedCycleUpdate = opsNeedCycleUpdate.front();
        opsNeedCycleUpdate.erase(opsNeedCycleUpdate.begin());
        auto taskOp = opNeedCycleUpdate.first;
        auto newStartCycle = opNeedCycleUpdate.second;

        auto taskId = getUniqueID(taskOp.getOperation());
        auto previousStartCycle = _operationStartCycle[taskId];

        if (newStartCycle > previousStartCycle) {
            auto delay = newStartCycle - previousStartCycle;
            auto newEndCycle = _operationEndCycle[taskId] + delay;

            // update cycle for current task
            _operationStartCycle[taskId] = newStartCycle;
            _operationEndCycle[taskId] = newEndCycle;
            // We only add start cycle time as time stamp for updated cycle information
            _orderedTimeStamp.insert(newStartCycle);
            taskOp->setAttr(cycleBegin, getIntAttr(taskOp->getContext(), newStartCycle));
            taskOp->setAttr(cycleEnd, getIntAttr(taskOp->getContext(), newEndCycle));

            // update cycle by explicit dependency
            for (auto consumer : _taskConsumerMapOriginal[taskOp.getOperation()]) {
                auto consumerId = getUniqueID(consumer);
                auto consumerStart = _operationStartCycle[consumerId];

                if (consumerStart < newEndCycle) {
                    _log.trace("Update start cycle from {0} to {1} for consumer task {2}", consumerStart, newEndCycle,
                               consumerId);
                    opsNeedCycleUpdate.push_back(std::make_pair(_orderedTasks[consumerId], newEndCycle));
                }
            }

            // Update cycle by implicit dependency between tasks of same executor. Note that dma task maybe in multi
            // queues.
            auto dmaTaskQueueTypes = getDMATaskQueueType(taskOp);
            if (dmaTaskQueueTypes.hasValue()) {
                for (auto& queueType : dmaTaskQueueTypes.getValue()) {
                    auto nextOpNeedCycleUpdate = updateCycleStartTimeDueToNativeExecutorDependency(
                            taskOp, newEndCycle, _orderedTasksByCycleStart[queueType]);
                    if (nextOpNeedCycleUpdate.hasValue()) {
                        opsNeedCycleUpdate.push_back(nextOpNeedCycleUpdate.getValue());
                    }
                }
            } else {
                auto taskQueueType = getTaskQueueType(taskOp);
                auto nextOpNeedCycleUpdate = updateCycleStartTimeDueToNativeExecutorDependency(
                        taskOp, newEndCycle, _orderedTasksByCycleStart[taskQueueType]);
                if (nextOpNeedCycleUpdate.hasValue()) {
                    opsNeedCycleUpdate.push_back(nextOpNeedCycleUpdate.getValue());
                }
            }
        }
    }
}

bool CycleBasedBarrierScheduler::attemptToScheduleTaskWithAvailableBarrier(VPURT::TaskOp taskOp) {
    mlir::Operation* task = taskOp.getOperation();
    size_t uniqueID = getUniqueID(task);
    size_t startCycle = _operationStartCycle[uniqueID];
    size_t taskBarrierResourceRequirement = _barrierResourceUtilizationMap[task];

    // If the internal scheduler current time is beyond the task's start cycle, it means that it was not possible to
    // start the task on time because the barrier resource was not available. Therefore, the assigned cycle start time
    // of the task by the feasible memory scheduler needs to be updated to reflect the delay.
    if (_currentTime > startCycle) {
        updateCycleStartTime(taskOp, _currentTime);
    }

    // If the internal scheduler time is greater than the cycle start time of the task and there is a free barrier
    // resource then this tasks can be scheduled.
    if ((_currentTime >= startCycle) &&
        isBarrierResourceAvailable(taskBarrierResourceRequirement, task, _heap.empty())) {
        auto scheduleSuccess = scheduleTask(task, taskBarrierResourceRequirement);
        VPUX_THROW_UNLESS(scheduleSuccess == true, "Failed to schedule task ID {0}", getUniqueID(task));

        _log.trace("Populating the scheduled tasks list with the relevant scheduling information for task ID",
                   getUniqueID(task));

        populateScheduledTasks(task);
        size_t taskEndTime = _operationEndCycle[uniqueID];

        _log.trace("Task ID {0} end time is {1}, pushing to heap", getUniqueID(task), taskEndTime);
        pushToScheduleTimeHeap(HeapElement(task, taskEndTime));
        // The end cycle time will be added only if a task is scheduled successfully. Because we know once the task is
        // scheduled, it must finish at its end time. It avoids to add redundant time stamp.
        _orderedTimeStamp.insert(taskEndTime);
        return true;
    }

    return false;
}

bool CycleBasedBarrierScheduler::performCycleBasedSchedulingTaskLoop() {
    _log.trace("Performing main scheduling task loop");
    _log = _log.nest();

    // cycle-start ordered loop
    // schedule NCE, DMA and UPA task in parallel
    _currentTime = 0;
    const auto module = _func->getParentOfType<mlir::ModuleOp>();
    auto dmaPortCount = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).count();
    size_t indNce, indUpa, indAct;
    SmallVector<size_t> indDmas(dmaPortCount, 0);
    indNce = indUpa = indAct = 0;
    // keeps looping until the index for each executor reaches the end, thus all tasks have been scheduled then
    SmallVector<TaskQueueType> dmaQueues(dmaPortCount, {VPU::ExecutorKind::DMA_NN, -1});
    constexpr TaskQueueType nceQueue{VPU::ExecutorKind::NCE, 0};
    constexpr TaskQueueType upaQueue{VPU::ExecutorKind::SHAVE_UPA, 0};
    constexpr TaskQueueType actQueue{VPU::ExecutorKind::SHAVE_ACT, 0};
    for (auto dmaIndex : irange(dmaQueues.size())) {
        dmaQueues[dmaIndex].index = checked_cast<int64_t>(dmaIndex);
    }

    auto checkDmaQueues = [&]() {
        for (const auto dmaQueueIndex : irange(dmaQueues.size())) {
            const auto& indDma = indDmas[dmaQueueIndex];
            if (indDma < _orderedTasksByCycleStart[dmaQueues[dmaQueueIndex]].size()) {
                return true;
            }
        }
        return false;
    };
    while (indNce < _orderedTasksByCycleStart[nceQueue].size() || checkDmaQueues() ||
           indUpa < _orderedTasksByCycleStart[upaQueue].size() || indAct < _orderedTasksByCycleStart[actQueue].size()) {
        _log.trace("Current time is {0}", _currentTime);
        _schedulingTasksInEachLoop.clear();
        // NCE task
        if (indNce < _orderedTasksByCycleStart[nceQueue].size()) {
            vpux::VPURT::TaskOp taskOp = _orderedTasksByCycleStart[nceQueue][indNce];
            if (attemptToScheduleTaskWithAvailableBarrier(taskOp)) {
                indNce++;
            }
        }

        // DMA task
        for (auto dmaQueueIndex : irange(dmaQueues.size())) {
            auto& indDma = indDmas[dmaQueueIndex];
            if (indDma < _orderedTasksByCycleStart[dmaQueues[dmaQueueIndex]].size()) {
                vpux::VPURT::TaskOp taskOp = _orderedTasksByCycleStart[dmaQueues[dmaQueueIndex]][indDma];
                // task has been scheduled in other dma queue, just skip it
                if (_taskScheduleStatus.find(taskOp) != _taskScheduleStatus.end() &&
                    _taskScheduleStatus[taskOp] == true) {
                    indDma++;
                } else if (attemptToScheduleTaskWithAvailableBarrier(taskOp)) {
                    _taskScheduleStatus[taskOp] = true;
                    indDma++;
                }
            }
        }

        // SHAVE UPA task
        if (indUpa < _orderedTasksByCycleStart[upaQueue].size()) {
            vpux::VPURT::TaskOp taskOp = _orderedTasksByCycleStart[upaQueue][indUpa];
            if (attemptToScheduleTaskWithAvailableBarrier(taskOp)) {
                indUpa++;
            }
        }

        // SHAVE ACT task
        if (indAct < _orderedTasksByCycleStart[actQueue].size()) {
            vpux::VPURT::TaskOp taskOp = _orderedTasksByCycleStart[actQueue][indAct];
            if (attemptToScheduleTaskWithAvailableBarrier(taskOp)) {
                indAct++;
            }
        }

        // When a task is scheduled to start, it will be removed from the consumer list of its wait barrier.
        // But We can't update the wait barrier immediately when populating the scheduling task, because the scheduling
        // loop over executor is in sequential. For example, if we update barrier for NCE before scheduling DMA in a
        // loop, some barriers could be free for use but it shouldn't happen because DMA should look at the same barrier
        // status as NCE.
        for (auto taskID : _schedulingTasksInEachLoop) {
            // Update consumer list of wait barrier when task is scheduled
            for (const auto& waitBarrier : _configureTaskOpWaitBarrier[taskID].set_bits()) {
                _barrierResourceState.updateBarrierConsumer(_orderedTasks[taskID].getOperation(),
                                                            _physicalID[waitBarrier]);
            }

            vpux::VPURT::TaskOp taskOp = _orderedTasks[taskID];
            // Set scheduling number
            taskOp->setAttr(schedulingNumberAttrName, getIntAttr(taskOp->getContext(), _schedulingOrder.size()));
            _schedulingOrder.push_back(taskID);
        }

        if (!_orderedTimeStamp.empty()) {
            // move up the schedule time to the next time stamp
            auto itr = _orderedTimeStamp.begin();
            _currentTime = *itr;
            _orderedTimeStamp.erase(itr);

            if (!_heap.empty()) {
                std::pop_heap(_heap.begin(), _heap.end(), MinHeapOrdering());
                HeapElement topElem = _heap.back();
                while ((!_heap.empty()) && (topElem._time <= _currentTime)) {
                    // since operation is now complete update the schedule
                    _log.trace("Unscheduling task ID {0}", getUniqueID(topElem._op));
                    auto unScheduleSucess = unScheduleTask(topElem._op);
                    VPUX_THROW_UNLESS(unScheduleSucess == true,
                                      "Failed to unschedule task ID {0}, unable to find the task's update barrier in "
                                      "the active barrier table",
                                      getUniqueID(topElem._op));
                    _heap.pop_back();
                    if (!_heap.empty()) {
                        topElem = _heap.back();
                    }
                }
            }
        } else {
            _log.trace("The schedule is not feasible, exiting...");
            return false;
        }
    }
    _log.trace("Finished performing main scheduling task loop");
    _log = _log.unnest();

    return true;
}

const CycleBasedBarrierResourceState& CycleBasedBarrierScheduler::barrierResourceState() const {
    return _barrierResourceState;
}

const CycleBasedBarrierScheduler::barrierInfo& CycleBasedBarrierScheduler::getBarrierInfo(mlir::Operation* op) const {
    auto itr = _barrierMap.find(op);
    VPUX_THROW_UNLESS(itr != _barrierMap.end(), "Could not find the operation in the active barrier map");
    return itr->second;
}

bool CycleBasedBarrierScheduler::unScheduleTask(mlir::Operation* op) {
    auto itr = _barrierMap.find(op);
    if (itr == _barrierMap.end()) {
        return false;
    }
    const barrierInfo& binfo = itr->second;
    auto unassignBarrierSlots =
            _barrierResourceState.unassignBarrierSlots(binfo._barrierIndex, binfo._producerSlotCount, op);

    VPUX_THROW_UNLESS(unassignBarrierSlots == true, "Failed to deallocate slots in the barrier index {0}",
                      binfo._barrierIndex);
    _barrierMap.erase(itr);
    return true;
}

bool CycleBasedBarrierScheduler::scheduleTask(mlir::Operation* op, const size_t producerSlotRequirement) {
    auto taskID = getUniqueID(op);
    _log.trace("Scheduling task ID {0}", getUniqueID(op));

    if (_barrierMap.find(op) != _barrierMap.end()) {
        return false;
    }

    // Find the latest wait barrier for current task.
    size_t latestWaitBarrier = 0;
    auto taskWaitMap = _configureTaskOpWaitBarrier[taskID].set_bits();
    for (const auto& waitBarrier : taskWaitMap) {
        if (latestWaitBarrier < waitBarrier) {
            latestWaitBarrier = waitBarrier;
        }
    }

    // Assume we will create a new barrier
    size_t barrierVirtualID = _configureBarrierOpWaitTask.size();
    size_t barrierPhysicalID =
            _barrierResourceState.assignBarrierSlots(producerSlotRequirement, op, latestWaitBarrier, barrierVirtualID,
                                                     _configureBarrierOpWaitTask, _configureBarrierOpUpdateTask);

    if (barrierVirtualID < _configureBarrierOpWaitTask.size()) {
        _log.trace("Use existing virtual barrier {0}", barrierVirtualID);
    } else {
        _log.trace("Create new virtual barrier {0}", barrierVirtualID);
        _physicalID.push_back(barrierPhysicalID);

        llvm::BitVector newBarrierProducers;
        newBarrierProducers.resize(static_cast<unsigned>(_taskCount));
        llvm::BitVector newBarrierConsumers;
        newBarrierConsumers.resize(static_cast<unsigned>(_taskCount));

        _configureBarrierOpWaitTask.push_back(newBarrierProducers);
        _configureBarrierOpUpdateTask.push_back(newBarrierConsumers);
    }

    // Update barrier and task dependency map
    _configureBarrierOpWaitTask[barrierVirtualID].set(static_cast<unsigned>(taskID));
    _configureTaskOpUpdateBarrier[taskID].set(static_cast<unsigned>(barrierVirtualID));

    for (auto task : _taskConsumerMapOriginal[op]) {
        auto consumerTaskID = getUniqueID(task);
        _configureBarrierOpUpdateTask[barrierVirtualID].set(static_cast<unsigned>(consumerTaskID));
        _configureTaskOpWaitBarrier[consumerTaskID].set(static_cast<unsigned>(barrierVirtualID));
    }

    _barrierMap.insert(std::make_pair(op, barrierInfo(barrierPhysicalID, producerSlotRequirement)));

    // Update cycle if consumers start ealier than producers finish because of barrier sharing
    size_t lastEnd = _operationEndCycle[getUniqueID(op)];
    auto waitMap = _configureBarrierOpWaitTask[barrierVirtualID].set_bits();
    if (!waitMap.empty()) {
        for (const auto& producer : waitMap) {
            if (lastEnd < _operationEndCycle[producer]) {
                lastEnd = _operationEndCycle[producer];
            }
        }
    }

    auto updateMap = _configureBarrierOpUpdateTask[barrierVirtualID].set_bits();
    if (!updateMap.empty()) {
        for (const auto& consumer : updateMap) {
            if (lastEnd > _operationStartCycle[consumer]) {
                updateCycleStartTime(_orderedTasks[consumer], lastEnd);
            }
        }
    }

    _log.trace("Finished scheduling task {0}", getUniqueID(op));
    return true;
}

bool CycleBasedBarrierScheduler::isBarrierResourceAvailable(const size_t producerSlotRequirement, mlir::Operation* task,
                                                            bool scheduledTasksAllFinished) {
    size_t latestWaitBarrier = 0;
    auto taskID = getUniqueID(task);
    auto waitMap = _configureTaskOpWaitBarrier[taskID].set_bits();
    for (const auto& waitBarrier : waitMap) {
        if (latestWaitBarrier < waitBarrier) {
            latestWaitBarrier = waitBarrier;
        }
    }

    auto itr = _barrierResourceState.findUnusedBarrierWithAvailableSlots(producerSlotRequirement);
    if (itr == _barrierResourceState._globalAvailableProducerSlots.end()) {
        itr = _barrierResourceState.findBarrierWithMinimumCycleDelay(producerSlotRequirement, latestWaitBarrier, task,
                                                                     _configureBarrierOpWaitTask,
                                                                     _configureBarrierOpUpdateTask);
    }

    if (itr != _barrierResourceState._globalAvailableProducerSlots.end()) {
        return true;
    }

    if (scheduledTasksAllFinished) {
        return _barrierResourceState.createUnusedBarrierByAdjustingConsumer(
                _configureBarrierOpWaitTask, _configureBarrierOpUpdateTask, _configureTaskOpWaitBarrier);
    }

    return false;
}

size_t CycleBasedBarrierScheduler::getUniqueID(mlir::Operation* op) {
    return checked_cast<size_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op)->getAttr("uniqueId").cast<mlir::IntegerAttr>().getInt());
}

void CycleBasedBarrierScheduler::assignTaskUniqueIds() {
    _orderedTasks.clear();
    int64_t uniqueId = 0;
    auto assignUniqueID = [&](VPURT::TaskOp taskOp) {
        taskOp->setAttr(uniqueIdAttrName, getIntAttr(taskOp->getContext(), uniqueId++));
    };

    _func.walk([&](VPURT::TaskOp taskOp) {
        assignUniqueID(taskOp);
        _orderedTasks.push_back(taskOp);
    });
    _taskCount = uniqueId;
}

// This function returns the number of producers to a barrier.
// On VPU H/W, a NCE task is executed across multiple DPUs via workloads descriptors (known as variants).
// Each variant must update the barrier to signal that is is complete.
// An NCE task may have up 50 workloads descriptors (which are generated in the NCE DPU workloads pass).
// Therefore, the number of variants must be retrieved here as they will all update a barrier and
// contribute to the 256 producer limit that a barrier has.
// A DMA/UPA does not have variants, therefore they always just requires 1 producer slot to a barrier.
size_t CycleBasedBarrierScheduler::countProducerTasksToBarrier(VPURT::TaskOp op) {
    if (op.getExecutorKind() == VPU::ExecutorKind::NCE) {
        auto innerTaskOp = op.getInnerTaskOp();
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerTaskOp)) {
            innerTaskOp = clusterTilingOp.getInnerTaskOp();
        }
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(innerTaskOp);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }
    if (op.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        auto innerTaskOp = op.getInnerTaskOp();
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerTaskOp)) {
            innerTaskOp = clusterTilingOp.getInnerTaskOp();
        }
        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(innerTaskOp);
        VPUX_THROW_UNLESS(swKernelOp != nullptr, "Could not cast to SwKernel task");
        auto swKernelRun = swKernelOp.body().getOps<VPUIP::SwKernelRun>();
        return std::distance(swKernelRun.begin(), swKernelRun.end());
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::DMA_NN || op.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
        return 1;
    }

    VPUX_THROW("This operation does not run on hardware");
}

// This function creates a table storing the number of producers to a barrier that a task requires.
void CycleBasedBarrierScheduler::createTaskBarrierResourceUtilityTable() {
    _log = _log.nest();
    for (auto& task : _orderedTasks) {
        auto barrierResouceUtilization = countProducerTasksToBarrier(task);
        _log.trace("Task {0} requires {1} barrier producer slots", getUniqueID(task.getOperation()),
                   barrierResouceUtilization);
        _barrierResourceUtilizationMap.insert(std::make_pair(task, barrierResouceUtilization));
    }
    _log = _log.unnest();
}

void CycleBasedBarrierScheduler::init(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    _log.trace("Feasible barrier scheduler initialization");
    _log = _log.nest();

    // As the background, DMA ops are pushed as a list to firmware, they are scheduled in that order and also finish in
    // that same exact order. This is true when we use the same DMA engine, but no longer when multiple engines are
    // enabled. In that case, DMA ops will be pushed into different lists according to their port values, and there's no
    // clear way to ensure execution serialization between DMA ops which used different engines. So the DMA related
    // optimization is enabled only when device has one DMA port.
    const auto module = _func->getParentOfType<mlir::ModuleOp>();
    _enableDMAOptimization = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).count() == 1;

    _barrierCount = numberOfBarriers;
    _slotsPerBarrier = maxProducersPerBarrier;

    // Assign unique IDs to tasks
    _log.trace("Assigning unique IDs to tasks");
    assignTaskUniqueIds();

    // Get cycle information
    getPerTaskStartAndEndCycleTime();

    // Optimize the original IR dependency
    _log.trace("Saving the original IR dependency");
    optimizeIRDependency();

    // Store per-task barrier producer utilization, i.e. the number of workloads
    _log.trace("Collating per task, the barrier resource requirements");
    createTaskBarrierResourceUtilityTable();

    _log.trace("Initializing the barrier resource upper state i.e. maximum barrier and maximum producers per barrier");
    _barrierResourceState.init(numberOfBarriers, maxProducersPerBarrier, _orderedTasks, _orderedTasksByCycleStart,
                               _taskConsumerMapOriginal);

    // it's a vector where every index refers to a task and the llvm::BitVector at each index is meant to
    // store the barrier that either update or wait the task.
    // We don't know the number of scheduled barriers at initialization stage. But it will not exceed the number of
    // tasks. So we resize llvm::BitVector with the number of tasks.
    _configureTaskOpWaitBarrier.resize(_taskCount);
    for (auto& wait : _configureTaskOpWaitBarrier) {
        wait.resize(static_cast<unsigned>(_taskCount));
    }

    _configureTaskOpUpdateBarrier.resize(_taskCount);
    for (auto& update : _configureTaskOpUpdateBarrier) {
        update.resize(static_cast<unsigned>(_taskCount));
    }

    _log = _log.unnest();
}

bool CycleBasedBarrierScheduler::generateScheduleWithBarriers() {
    _log.trace("Starting to generate a schedule with {0} barriers", _barrierCount);
    _log = _log.nest();

    // Scheduling loop, loop until all tasks are scheduled
    bool scheduleSuccess = performCycleBasedSchedulingTaskLoop();
    VPUX_THROW_UNLESS(scheduleSuccess == true, "Failed to generate a valid schedule including barriers.");

    // Insert barriers in the IR based on the output of the list scheduler
    _log.trace("Inserting barriers in the IR");
    insertBarriersinIR();
    _log = _log.unnest();

    _log.trace("Finished generating a schedule with barriers");

    return scheduleSuccess;
}

void CycleBasedBarrierScheduler::reorderIR() {
    // reorder barrier by id
    VPURT::DeclareVirtualBarrierOp preBarrier = nullptr;
    for (auto& curBarrier : _orderedBarrier) {
        if (preBarrier) {
            curBarrier->moveAfter(preBarrier);
        }
        preBarrier = curBarrier;
    }

    // reorder task by scheduling number
    VPURT::TaskOp preTask = nullptr;
    for (auto& ind : _schedulingOrder) {
        auto curTask = _orderedTasks[ind];
        if (preTask) {
            curTask->moveAfter(preTask);
        }
        preTask = curTask;
    }
}

void CycleBasedBarrierScheduler::insertBarriersinIR() {
    size_t barrierCount = _configureBarrierOpWaitTask.size();
    mlir::OpBuilder builder(_func.getBody());

    _log = _log.nest();

    for (size_t barrierTaskId = 0; barrierTaskId < barrierCount; barrierTaskId++) {
        auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(_orderedTasks[0]->getLoc());
        newBarrier->setAttr(virtualIdAttrName, getIntAttr(newBarrier->getContext(), barrierTaskId));
        _orderedBarrier.push_back(newBarrier);
    }

    _log.trace("Barrier scheduling complete");

    _log.trace("Task number is {0}", _orderedTasks.size());
    _log.trace("Barrier number is {0} before optimization", _orderedBarrier.size());
    _log.trace("Removing redundant dependencies");
    removeRedundantDependencies();

    _log.trace("Removing redundant barriers");
    removeRedundantBarriers(false);

    _log.trace("Removing redundant dependencies");
    removeRedundantDependencies();

    for (size_t ind = 0; ind < _configureBarrierOpUpdateTask.size(); ind++) {
        _log.trace("Virtual Barrier ID {0} has {1} consumers", ind, _configureBarrierOpUpdateTask[ind].count());
    }

    for (size_t ind = 0; ind < _configureBarrierOpWaitTask.size(); ind++) {
        _log.trace("Virtual Barrier ID {0} has {1} producers", ind, _configureBarrierOpWaitTask[ind].count());
    }

    _log.trace("Starting to add barriers into the IR");
    for (size_t ind = 0; ind < _configureBarrierOpWaitTask.size(); ind++) {
        auto& barrierOp = _orderedBarrier[ind];
        auto waitMap = _configureBarrierOpWaitTask[ind].set_bits();
        auto updateMap = _configureBarrierOpUpdateTask[ind].set_bits();
        if (waitMap.empty() || updateMap.empty()) {
            barrierOp->dropAllUses();
            barrierOp.erase();
            barrierOp = nullptr;
        } else {
            for (const auto& user : waitMap) {
                auto taskOp = _orderedTasks[user];

                VPUX_THROW_UNLESS(taskOp != NULL, "Invalid task");
                VPUX_THROW_UNLESS(barrierOp.barrier() != NULL, "Invalid barrier");
                _log.trace("Adding Barrier ID {0} as an update barrier for operation {1}",
                           barrierOp->getAttr(virtualIdAttrName), getUniqueID(taskOp));
                taskOp.updateBarriersMutable().append(barrierOp.barrier());
            }

            for (const auto& user : updateMap) {
                auto taskOp = _orderedTasks[user];
                VPUX_THROW_UNLESS(taskOp != NULL, "Invalid task");
                VPUX_THROW_UNLESS(barrierOp.barrier() != NULL, "Invalid barrier");
                _log.trace("Adding Barrier ID {0} as an wait barrier for operation {1}",
                           barrierOp->getAttr(virtualIdAttrName), getUniqueID(taskOp));
                taskOp.waitBarriersMutable().append(barrierOp.barrier());
            }
        }
    }
    _log.trace("Finished adding barriers into the IR");
    _log = _log.unnest();

    _orderedBarrier.erase(std::remove(_orderedBarrier.begin(), _orderedBarrier.end(), nullptr), _orderedBarrier.end());
    _log.trace("Barrier number is {0} after optimization", _orderedBarrier.size());
}

void CycleBasedBarrierScheduler::removeVirtualBarriers() {
    _log.trace("Removing the original declare virtual barrier ops");
    _func->walk([](VPURT::TaskOp op) {
        op.updateBarriersMutable().clear();
        op.waitBarriersMutable().clear();
    });

    _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
        op->dropAllUses();
        op.erase();
    });
}

void CycleBasedBarrierScheduler::clearTemporaryAttributes() {
    _func->walk([](VPURT::TaskOp op) {
        op->removeAttr(uniqueIdAttrName);
        op->removeAttr(virtualIdAttrName);
        op->removeAttr(schedulingNumberAttrName);
    });
}

bool CycleBasedBarrierScheduler::performRuntimeSimulation() {
    bool success = true;

    _log.trace("Starting runtime simulation");
    reorderIR();
    if (_orderedBarrier.size()) {
        // run simulation
        VPURT::BarrierSimulator barrierSim(_func);
        VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");
        success = mlir::succeeded(barrierSim.simulateBarriers(_log.nest()));
    }

    VPUX_THROW_UNLESS(success, "Barrier simulation was not successful");
    _log.trace("Barrier simulation result is {0} with upperbound {1}", success, _barrierCount);
    return success;
}

/// @brief check if two barriers can be merged
/// @details two barriers A and B can be merged if
/// 1. any producer of barrier A controls any consumer of barrier B
/// 2. any producer of barrier B controls any consumer of barrier A
bool CycleBasedBarrierScheduler::canBarriersBeMerged(size_t barrier1, size_t barrier2) {
    const auto& producers1 = _configureBarrierOpWaitTask[barrier1];
    const auto& consumers1 = _configureBarrierOpUpdateTask[barrier1];
    const auto& producers2 = _configureBarrierOpWaitTask[barrier2];
    const auto& consumers2 = _configureBarrierOpUpdateTask[barrier2];

    // The final dma port is assigned in unrolling stage. So we can't utilize the native dependency within a DMA engine,
    // which means two barrers can be merged only if they have exactly same consumers.
    if (!_enableDMAOptimization) {
        return (consumers1 == consumers2);
    }

    if (producers1.set_bits().empty() || consumers1.set_bits().empty() || producers2.set_bits().empty() ||
        consumers2.set_bits().empty()) {
        return false;
    }

    // In theory we need to check the control path from a barrier's producer to another barrier's consumer for every two
    // barriers. However that almost requires a check between every two tasks, which might be time consuming and even
    // cause stack overflow because the check is implemented as a recursive function. So currently we will simply the
    // check only for DMA. For example, we can merge following two barriers by considering the native dependency within
    // a DMA engine.
    /*
       DMA-0              DPU-0
         |                  |
       Barrier-0          Barrier-1
         |               /         \
       DPU-1           DPU-1      DMA-1

       merge into a single barrier

       DMA-0    DPU-0
         \        /
          Barrier-0
         /        \
       DPU-1    DMA-1
    */

    for (auto prod : producers1.set_bits()) {
        for (auto cons : consumers2.set_bits()) {
            if ((!consumers1.test(cons)) && ((_orderedTasks[prod].getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
                                             (_orderedTasks[cons].getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
                                             (!_orderedTasks[prod]->isBeforeInBlock(_orderedTasks[cons])))) {
                return false;
            }
        }
    }

    for (auto prod : producers2.set_bits()) {
        for (auto cons : consumers1.set_bits()) {
            if ((!consumers2.test(cons)) && ((_orderedTasks[prod].getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
                                             (_orderedTasks[cons].getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
                                             (!_orderedTasks[prod]->isBeforeInBlock(_orderedTasks[cons])))) {
                return false;
            }
        }
    }

    return true;
}

// If two barriers have same consumers, they can be merged
// If a barrier has no producers, it can be removed
// If a barrier only has DMA producers and consumers, it can be removed
//
// Notes: there is a limitation about the producer number of invariants and variants of a barrier
// The limitation is not related to HW capabilities or FIFO depth, but to the fact that the runtime needs to know when a
// workload is completed, in order to replace it with another one in NN CMX.
//
// Since there's no other efficient feedback mechanism from DPU/SNN to LNN, LNN monitors the barrier production of DPU
// tasks and recycles the storage when the corresponding barrier gets produced. The limitation comes from how many
// invariants/variants can be stored in NN CMX at the same time. For single cluster inferences these counts are 64/512,
// while for 4-cluster inferences 128/512. If too many invariants/variants contribute to the same barrier, the runtime
// will not receive the confirmation that it may recycle the storage to bring in the next workloads, hence the deadlock.
//
// Since the storage area is double buffered, and the workloads in question may start at any index in the buffer, it's
// only safe for at most <storage_size / 2 + 1> consecutive invariants/variants to produce the same barrier. So finally,
// the limits are:
//
// On single cluster:
//   32 + 1 invariants
//   256 + 1 variants
// On 4 clusters:
//   64 + 1 invariants
//   256 + 1 variants
//
void CycleBasedBarrierScheduler::removeRedundantBarriers(bool optimizeIRDependency) {
    if (_enableDMAOptimization) {
        // Remove explicit barrier dependency between DMAs before merging barriers
        // 1) if a barrier only has DMAs with same port as its producer, remove all DMAs with same dma port from its
        // consumers
        // 2) if a barrier only has DMAs with same port as its consumer, remove all DMAs with same dma port from
        // its producers
        for (size_t ind = 0; ind < _configureBarrierOpWaitTask.size(); ind++) {
            auto& producers = _configureBarrierOpWaitTask[ind];
            auto& consumers = _configureBarrierOpUpdateTask[ind];
            if (!producers.set_bits().empty()) {
                Optional<TaskQueueType> producerDMATaskType = None;
                bool barrierOnlyProducedByDMA = true;
                for (auto prod : producers.set_bits()) {
                    if (_orderedTasks[prod].getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
                        barrierOnlyProducedByDMA = false;
                        break;
                    } else {
                        auto curDMATaskType = getTaskQueueType(_orderedTasks[prod]);
                        if (!producerDMATaskType.hasValue()) {
                            producerDMATaskType = curDMATaskType;
                        } else if (curDMATaskType != producerDMATaskType.getValue()) {
                            barrierOnlyProducedByDMA = false;
                            break;
                        }
                    }
                }
                if (barrierOnlyProducedByDMA) {
                    for (auto cons : consumers.set_bits()) {
                        if (getTaskQueueType(_orderedTasks[cons]) == producerDMATaskType.getValue()) {
                            consumers.reset(cons);
                            _configureTaskOpWaitBarrier[cons].reset(ind);
                        }
                    }
                }
            }

            if (!consumers.set_bits().empty()) {
                bool barrierOnlyConsumedByDMA = true;
                Optional<TaskQueueType> consumerDMATaskType;
                for (auto cons : consumers.set_bits()) {
                    if (_orderedTasks[cons].getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
                        barrierOnlyConsumedByDMA = false;
                        break;
                    } else {
                        auto curDMATaskType = getTaskQueueType(_orderedTasks[cons]);
                        if (!consumerDMATaskType.hasValue()) {
                            consumerDMATaskType = curDMATaskType;
                        } else if (curDMATaskType != consumerDMATaskType.getValue()) {
                            barrierOnlyConsumedByDMA = false;
                            break;
                        }
                    }
                }
                if (barrierOnlyConsumedByDMA) {
                    for (auto prod : producers.set_bits()) {
                        if (getTaskQueueType(_orderedTasks[prod]) == consumerDMATaskType.getValue()) {
                            producers.reset(prod);
                            _configureTaskOpUpdateBarrier[prod].reset(ind);
                        }
                    }
                }
            }
        }
    }

    for (size_t ind = 0; ind < _configureBarrierOpUpdateTask.size(); ind++) {
        auto& consumers = _configureBarrierOpUpdateTask[ind];
        if (!(consumers.set_bits().empty())) {
            auto ind1 = ind + 1;
            for (; ind1 < _configureBarrierOpUpdateTask.size(); ind1++) {
                if (canBarriersBeMerged(ind, ind1)) {
                    _log.trace("Found barrier {0} and {1} have same consumers", ind, ind1);
                    auto& producers = _configureBarrierOpWaitTask[ind1];
                    auto& consumers1 = _configureBarrierOpUpdateTask[ind1];
                    size_t variantsCount = 0;
                    size_t invariantsCount = 0;
                    for (auto oldProducer : _configureBarrierOpWaitTask[ind].set_bits()) {
                        variantsCount += countProducerTasksToBarrier(_orderedTasks[oldProducer]);
                        invariantsCount++;
                    }
                    for (auto newProducer : producers.set_bits()) {
                        variantsCount += countProducerTasksToBarrier(_orderedTasks[newProducer]);
                        invariantsCount++;
                    }

                    auto module = _func->getParentOfType<mlir::ModuleOp>();
                    auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
                    auto numClusters = nceOp.count();
                    // ClusterTiling is not Unrolled yet. So set invariants limits as 64/4 on 4 clusters.
                    size_t invariantsLimits = (numClusters == 1) ? 32 : 16;
                    // When this function is used to optimize IR dependency, we don't need to consider the limits of
                    // invariants and variants
                    if (optimizeIRDependency ||
                        ((variantsCount <= _slotsPerBarrier) && (invariantsCount <= invariantsLimits))) {
                        _log.trace("Merge barrier {0} and {1}", ind, ind1);
                        for (auto task : producers.set_bits()) {
                            _configureBarrierOpWaitTask[ind].set(task);
                            _configureTaskOpUpdateBarrier[task].set(ind);
                        }
                        for (auto task : consumers1.set_bits()) {
                            _configureBarrierOpUpdateTask[ind].set(task);
                            _configureTaskOpWaitBarrier[task].set(ind);
                        }
                        producers.reset();
                        consumers1.reset();
                    }
                }
            }
        }
    }
}

// For two producers {a, b} of a barrier, if a depends on b then b isn't a necessary producer for this barrier
// For two consumers {a, b} of a barrier, if a depends on b then a isn't a necessary consumer for this barrier
void CycleBasedBarrierScheduler::removeRedundantDependencies() {
    for (size_t ind = 0; ind < _configureBarrierOpWaitTask.size(); ind++) {
        // producers
        auto& producers = _configureBarrierOpWaitTask[ind];
        SmallVector<unsigned> producersToRemove;
        for (auto prod = producers.set_bits_begin(); prod != producers.set_bits_end(); prod++) {
            auto prod1 = prod;
            prod1++;
            for (; prod1 != producers.set_bits_end(); prod1++) {
                if (doesPathExist(*prod1, *prod, false)) {
                    producersToRemove.push_back(*prod1);
                } else if (doesPathExist(*prod, *prod1, false)) {
                    producersToRemove.push_back(*prod);
                    break;
                }
            }
        }

        for (auto& producer : producersToRemove) {
            producers.reset(producer);
            _configureTaskOpUpdateBarrier[producer].reset(ind);
        }

        // consumers
        auto& consumers = _configureBarrierOpUpdateTask[ind];
        SmallVector<unsigned> consumersToRemove;
        for (auto cons = consumers.set_bits_begin(); cons != consumers.set_bits_end(); cons++) {
            auto cons1 = cons;
            cons1++;
            for (; cons1 != consumers.set_bits_end(); cons1++) {
                if (doesPathExist(*cons, *cons1, true)) {
                    consumersToRemove.push_back(*cons1);
                } else if (doesPathExist(*cons1, *cons, true)) {
                    consumersToRemove.push_back(*cons);
                    break;
                }
            }
        }

        for (auto& consumer : consumersToRemove) {
            consumers.reset(consumer);
            _configureTaskOpWaitBarrier[consumer].reset(ind);
        }
    }
}

// detect if op b depends on a
bool CycleBasedBarrierScheduler::doesPathExist(int64_t a, int64_t b, bool checkConsumer) {
    auto findPath = [&](int64_t task1, int64_t task2) {
        auto numa = _orderedTasks[task1]->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
        auto numb = _orderedTasks[task2]->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
        if (numa > numb)
            return false;

        if (numa == numb)
            return true;

        // DMAs which are scheduled later in the schedule naturally depend on DMAs which were scheduled previously
        // But for DPUs, we need to consider their dependency seperately according they are barrier's producers or
        // consumers Because if DPU task A is before B in execution list, that means A starts before B starts but
        // not gurantee A finishes after B finishes The reason is we have multiple DPUs to execute the workloads, if
        // A doesn't use all the DPUs then B could execute in parallel. And the finishing time depends on the
        // computation cost of individual workload. But the fact of A starts before B starts is good enough for
        // barrier's consumers because barrier only controls the starting of consumers
        //
        // UPA task also has multiple execution cores so we treat it the same way as DPU task.
        if (checkConsumer &&
            (getTaskQueueType(_orderedTasks[task1], false) == getTaskQueueType(_orderedTasks[task2], false)) &&
            _orderedTasks[task1].getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return true;
        } else if (_enableDMAOptimization && _orderedTasks[task1].getExecutorKind() == VPU::ExecutorKind::DMA_NN &&
                   getTaskQueueType(_orderedTasks[task1]) == getTaskQueueType(_orderedTasks[task2])) {
            return true;
        }
        auto updateBarriers = _configureTaskOpUpdateBarrier[task1];
        for (auto updateBarrier : updateBarriers.set_bits()) {
            auto barrierConsumers = _configureBarrierOpUpdateTask[updateBarrier];
            for (auto consumer : barrierConsumers.set_bits()) {
                if (consumer == task2)
                    return true;
            }
            for (auto consumer : barrierConsumers.set_bits()) {
                if (doesPathExist(consumer, task2, checkConsumer))
                    return true;
            }
        }
        if (_enableDMAOptimization && _orderedTasks[task1].getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
            auto taskQueueType = getTaskQueueType(_orderedTasks[task1]);
            auto itr = std::find(_orderedTasksByCycleStart[taskQueueType].begin(),
                                 _orderedTasksByCycleStart[taskQueueType].end(), _orderedTasks[task1]);
            VPUX_THROW_UNLESS(itr != _orderedTasksByCycleStart[taskQueueType].end(), "task {0} is not found in list",
                              task1);
            itr++;
            if (itr != _orderedTasksByCycleStart[taskQueueType].end()) {
                auto nextTaskID = getUniqueID((*itr).getOperation());
                if (doesPathExist(nextTaskID, task2, checkConsumer))
                    return true;
            }
        }
        return false;
    };

    auto path = std::make_tuple(a, b, checkConsumer);
    if (_pathLookUpTable.find(path) == _pathLookUpTable.end()) {
        _pathLookUpTable[path] = findPath(a, b);
    }
    return _pathLookUpTable[path];
}

/// @brief populate scheduled task into a list and print scheduling information, e.g. scheduled time and barrier
/// index
/// @param task scheduled task
void CycleBasedBarrierScheduler::populateScheduledTasks(mlir::Operation* task) {
    _log.trace("Populating the scheduling info for the scheduled task {0}", getUniqueID(task));
    _log = _log.nest();
    _log.trace("Get barrier info for task{0}", getUniqueID(task));
    const barrierInfo& binfo = getBarrierInfo(task);

    _log.trace("Task {0} is scheduled in time  {1}", getUniqueID(task), _currentTime);
    _log.trace("The task's barrier index is {0} and the slot count is {1}", binfo._barrierIndex,
               binfo._producerSlotCount);

    // Set scheduling number
    _log.trace("Assigning scheduling number {0} to the task {1} ", _schedulingOrder.size(), getUniqueID(task));
    _schedulingTasksInEachLoop.push_back(getUniqueID(task));
    _log = _log.unnest();
}

/// @brief get start/end cycle time for each task and sort
/// @details get the start and end cycle time generated by feasible memory scheduler. Populate tasks into different
/// lists according to executor type and sort them by start cycle time.
void CycleBasedBarrierScheduler::getPerTaskStartAndEndCycleTime() {
    _operationStartCycle.resize(_taskCount);
    _operationEndCycle.resize(_taskCount);

    _func.walk([&](VPURT::TaskOp taskOp) {
        auto opId = getUniqueID(taskOp.getOperation());
        auto dmaTaskQueueTypes = getDMATaskQueueType(taskOp);
        if (dmaTaskQueueTypes.hasValue()) {
            for (auto& queueType : dmaTaskQueueTypes.getValue()) {
                _orderedTasksByCycleStart[queueType].push_back(taskOp);
            }
        } else {
            auto taskQueueType = getTaskQueueType(taskOp);
            _orderedTasksByCycleStart[taskQueueType].push_back(taskOp);
        }

        if (taskOp->hasAttr(cycleBegin)) {
            auto currCycle = taskOp->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getValue().getSExtValue();
            _operationStartCycle[opId] = checked_cast<size_t>(currCycle);
            // We only add start cycle time as time stamp at the begining
            _orderedTimeStamp.insert(_operationStartCycle[opId]);
        } else {
            VPUX_THROW("TaskOp {0} was not assigned a start cycle time by the CMX memory scheduler", taskOp);
        }

        if (taskOp->hasAttr(cycleEnd)) {
            auto currCycle = taskOp->getAttr(cycleEnd).cast<mlir::IntegerAttr>().getValue().getSExtValue();
            _operationEndCycle[opId] = checked_cast<size_t>(currCycle);
        } else {
            VPUX_THROW("TaskOp {0} was not assigned an end cycle time by the CMX memory scheduler", taskOp);
        }
    });

    // execution order is decided by cycle start
    for (auto taskList : _orderedTasksByCycleStart) {
        std::sort(taskList.second.begin(), taskList.second.end(), startCycleTaskComparator());
    }
}
