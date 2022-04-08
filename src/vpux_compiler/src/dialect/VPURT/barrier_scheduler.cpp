//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"

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
// In the context of VPU hardware the number of slots for a DPU tasks are the DPU worklaods
// and for a DMA/UPA tasks it is 1.

BarrierScheduler::BarrierScheduler(mlir::FuncOp func, Logger log)
        : _barrierCount(),
          _slotsPerBarrier(),
          _barrierResourceState(),
          _inDegree(),
          _originalInDegree(),
          _heap(),
          _currentTime(0),
          _schedulableCandidates(),
          _processedTasks(),
          _priority(),
          _scheduledTasks(),
          _barrierAssociationTable(),
          _barrierResourceUtilizationMap(),
          _outputTasks(),
          _originalOutputOps(),
          _barrierMap(),
          _configureBarrierOpWaitMap(),
          _configureBarrierOpUpdateMap(),
          _configureTaskOpWaitMap(),
          _configureTaskOpUpdateMap(),
          _taskConsumerMapOriginal(),
          _log(log),
          _func(func),
          _taskCount(){};

void BarrierScheduler::populateTasksUpdateWaitBarrierMap(barrierWaitMapType& barrierOpWaitMap,
                                                         barrierUpdateMapType& barrierOpUpdateMap,
                                                         taskOpWaitMapType& taskOpWaitMap,
                                                         taskOpUpdateMapType& taskOpUpdateMap) {
    size_t virtualBarrierCount = barrierOpWaitMap.size();
    taskOpWaitMap.resize(_taskCount);
    for (auto& wait : taskOpWaitMap) {
        wait.resize((unsigned)virtualBarrierCount);
    }

    taskOpUpdateMap.resize(_taskCount);
    for (auto& update : taskOpUpdateMap) {
        update.resize((unsigned)virtualBarrierCount);
    }

    for (unsigned ind = 0; ind < barrierOpWaitMap.size(); ind++) {
        auto waitMap = barrierOpWaitMap[ind].set_bits();
        for (auto wait : waitMap) {
            taskOpUpdateMap[wait].set(ind);
        }
    }

    for (unsigned ind = 0; ind < barrierOpUpdateMap.size(); ind++) {
        auto updateMap = barrierOpUpdateMap[ind].set_bits();
        for (auto update : updateMap) {
            taskOpWaitMap[update].set(ind);
        }
    }
}

void BarrierScheduler::saveOriginalIRDependency() {
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

    // Compute in-degree and consumers of tasks
    const auto updateInDegreeAndConsumers = [&](VPURT::TaskOp taskOp) {
        size_t count = 0;
        for (const auto bar : taskOp.waitBarriers()) {
            auto iter = barrierOpUpdateWaitMap.find(bar.getDefiningOp());
            if (iter != barrierOpUpdateWaitMap.end()) {
                count += iter->second.first.size();
            } else {
                VPUX_THROW("barrier '{0}' not found", bar.getDefiningOp());
            }
        }
        _originalInDegree.insert(std::make_pair(taskOp, count));

        SmallVector<mlir::Operation*> consumers;
        for (const auto bar : taskOp.updateBarriers()) {
            auto iter = barrierOpUpdateWaitMap.find(bar.getDefiningOp());
            if (iter != barrierOpUpdateWaitMap.end()) {
                consumers.insert(consumers.end(), iter->second.second.begin(), iter->second.second.end());
            } else {
                VPUX_THROW("barrier '{0}' not found", bar.getDefiningOp());
            }
        }
        _taskConsumerMapOriginal.insert(std::make_pair(taskOp, consumers));

        if (consumers.empty())
            _originalOutputOps.insert(taskOp);
    };

    _func->walk([&](VPURT::TaskOp taskOp) {
        updateBarrierConfigs(taskOp);
    });

    _func->walk([&](VPURT::TaskOp taskOp) {
        updateInDegreeAndConsumers(taskOp);
    });

    // Remove the original virtual barriers, optimal barriers are inserted based on the generated schedule
    removeVirtualBarriers();

    _log.trace("Removed all the original declare virtual barrier ops");
}

void BarrierScheduler::pushToScheduleTimeHeap(const HeapElement& elem) {
    _heap.push_back(elem);
    std::push_heap(_heap.begin(), _heap.end(), MinHeapOrdering());
}

BarrierScheduler::HeapElement BarrierScheduler::popFromHeap() {
    std::pop_heap(_heap.begin(), _heap.end(), MinHeapOrdering());
    HeapElement elem = _heap.back();
    _heap.pop_back();
    return elem;
}

void BarrierScheduler::addTaskToCandidateSet(mlir::Operation* op) {
    if (_processedTasks.find(op) != _processedTasks.end()) {
        VPUX_THROW("Attempt to add a task to the schedulable candidates list that has been previously scheduled");
    }

    _schedulableCandidates.push_back(op);
    _processedTasks.insert(op);
}

void BarrierScheduler::addOutGoingOperationsToCandidateList(mlir::Operation* op) {
    _log.trace("Add outgoing operations to candidate list");
    _log = _log.nest();

    // Reduce indegree (number of incoming edges) for consumers of ready data ops
    // decrement the in-degree of &(*itr) and only add to candidate set
    // if the indegree is zero. This means this op is ready to be scheduled.

    auto opConsumers = getConsumerOps(op);

    SmallVector<mlir::Operation*>::iterator itr = opConsumers.begin();
    SmallVector<mlir::Operation*>::iterator itr_end = opConsumers.end();

    for (; itr != itr_end; ++itr) {
        // decrement the in-degree of &(*itr) and only add to candidate set
        // if the indegree is zero. This means this op is ready to be scheduled.

        mlir::Operation* op = (*itr);

        _log.trace("Decrementing the in-degree of operation {0}", getUniqueID(*itr));

        typename operationInDegreeType::iterator deg_itr = _inDegree.find(op);

        VPUX_THROW_UNLESS((deg_itr != _inDegree.end()) && (deg_itr->second > 0), "Invalid indegree");

        if (deg_itr->second == 1) {
            _log.trace("Adding operation {0} to candidate_list", getUniqueID(*itr));
            addTaskToCandidateSet(op);
            _log.trace("Erasing operation {0} from the in_degree table", getUniqueID(*itr));
            _inDegree.erase(deg_itr);
        } else {
            --(deg_itr->second);
        }
    }
    _log.trace("Finished adding outgoing operations to candidate list");
    _log = _log.unnest();
}

bool BarrierScheduler::performSchedulingTaskLoop() {
    _log.trace("Performing main scheduling task loop");
    _log = _log.nest();

    // Scheduling loop, loop until all output tasks are scheduled
    while (!_outputTasks.empty()) {
        schedulableTasksIteratorType taskItr = findSchedulableTask();

        if (isTaskInSchedulableCandidates(taskItr)) {
            // Found a schedulable task
            mlir::Operation* task = (*taskItr);
            _log.trace("Found a schedulable task ID {0}", getUniqueID(task));

            const size_t opDelay = 1;
            size_t taskBarrierResourceRequirement = _barrierResourceUtilizationMap[task];
            size_t taskEndTime = _currentTime + opDelay;

            _log.trace("Task ID {0} end time is {1}, pushing to heap", getUniqueID(task), taskEndTime);
            pushToScheduleTimeHeap(HeapElement(task, taskEndTime));

            _log.trace("Erasing Task ID {0} from the schedulable candidates");
            _schedulableCandidates.erase(taskItr);

            // schedule task
            auto scheduleSuccess = scheduleTask(task, taskBarrierResourceRequirement);
            VPUX_THROW_UNLESS(scheduleSuccess == true, "Failed to schedule task ID {0}", getUniqueID(task));

            _log.trace("Populating the scheduled tasks list with the relevant scheduling information for task ID",
                       getUniqueID(task));
            populateScheduledTasks(task);

            // decrease outputs tasks if output task is scheduled
            if (_outputTasks.find(task) != _outputTasks.end()) {
                _outputTasks.erase(task);
            }

        } else if (!_heap.empty()) {
            // no-op found so move up the schedule time to the smallest completion
            // time among the active operations
            HeapElement topElem = popFromHeap();

            VPUX_THROW_UNLESS(
                    _currentTime <= topElem._time,
                    "An error has occurred the _currentScheduleTime should not be less than time popped from the heap");

            _currentTime = topElem._time;
            // since operation is now complete update the schedule

            _log.trace("Unscheduling task ID {0}", getUniqueID(topElem._op));
            auto unScheduleSucess = unScheduleTask(topElem._op);

            VPUX_THROW_UNLESS(unScheduleSucess == true, "Failed to unschedule task ID {0}", getUniqueID(*taskItr));

            // since op has completed add all out-going ops to candidates
            _log.trace("Adding children tasks for task ID {0} to be candidates to schedule", getUniqueID(topElem._op));
            addOutGoingOperationsToCandidateList(topElem._op);
        } else {
            // schedule is not feasible
            _log.trace("The schedule is not feasible, exiting...");
            _schedulableCandidates.clear();
            return false;
        }
    }
    _log.trace("Finished performing main scheduling task loop");
    _log = _log.unnest();

    return true;
}

bool BarrierScheduler::isTaskInSchedulableCandidates(schedulableTasksIteratorType itr) const {
    return !(itr == _schedulableCandidates.end());
}

BarrierScheduler::schedulableTasksIteratorType BarrierScheduler::findSchedulableTask() {
    _log.trace("Looking for a scheduleable task");
    _log = _log.nest();

    schedulableTasksIteratorType itr = _schedulableCandidates.end();
    std::list<schedulableTasksIteratorType> readyList;

    _log.trace("There are {0} candidates", _schedulableCandidates.size());

    for (itr = _schedulableCandidates.begin(); itr != _schedulableCandidates.end(); ++itr) {
        _log.trace("The producerSlotRequirement for task {0} is {1}", getUniqueID(*itr),
                   _barrierResourceUtilizationMap[*itr]);

        if (isBarrierResourceAvailable(_barrierResourceUtilizationMap[*itr])) {
            _log.trace("Adding task {0} to the ready list", getUniqueID(*itr));
            readyList.push_back(itr);
        }
    }

    _log = _log.unnest();
    _log.trace("Finding the task with lowest priority in ready list");
    // find the one with lowest priority //
    if (!readyList.empty()) {
        size_t minPriority = std::numeric_limits<size_t>::max();
        for (auto ritr = readyList.begin(); ritr != readyList.end(); ++ritr) {
            size_t currentPriority = _priority[*(*ritr)];
            if (currentPriority < minPriority) {
                itr = *ritr;
                minPriority = currentPriority;
            }
        }
    }

    return itr;
}

const BarrierResourceState& BarrierScheduler::barrierResourceState() const {
    return _barrierResourceState;
}

const BarrierScheduler::barrierInfo& BarrierScheduler::getBarrierInfo(mlir::Operation* op) const {
    auto itr = _barrierMap.find(op);
    VPUX_THROW_UNLESS(itr != _barrierMap.end(), "Could not find the operation in the active barrier map");
    return itr->second;
}

bool BarrierScheduler::unScheduleTask(mlir::Operation* op) {
    auto itr = _barrierMap.find(op);
    if (itr == _barrierMap.end()) {
        return false;
    }
    const barrierInfo& binfo = itr->second;
    auto unassignBarrierSlots =
            _barrierResourceState.unassignBarrierSlots(binfo._barrierIndex, binfo._producerSlotCount);

    VPUX_THROW_UNLESS(unassignBarrierSlots == true, "Failed to deallocate slots in the barrier index {0}",
                      binfo._barrierIndex);
    _barrierMap.erase(itr);
    return true;
}

bool BarrierScheduler::scheduleTask(mlir::Operation* op, const size_t producerSlotRequirement) {
    _log.trace("Scheduling task ID {0}", getUniqueID(op));

    VPUX_THROW_UNLESS(isBarrierResourceAvailable(producerSlotRequirement) == true,
                      "Attempt to schedule task failed, failed to allocate barrier resource for task {0}}",
                      getUniqueID(op));

    if (_barrierMap.find(op) != _barrierMap.end()) {
        return false;
    }
    size_t barrierID = _barrierResourceState.assignBarrierSlots(producerSlotRequirement);
    _barrierMap.insert(std::make_pair(op, barrierInfo(barrierID, producerSlotRequirement)));

    _log.trace("Finished scheduling task {0}", getUniqueID(op));
    return true;
}

bool BarrierScheduler::isBarrierResourceAvailable(const size_t producerSlotRequirement) {
    return _barrierResourceState.hasBarrierWithAvailableSlots(producerSlotRequirement);
}

void BarrierScheduler::initializeBarrierResourceState(const size_t numberOfBarriers,
                                                      const size_t maxProducersPerBarrier) {
    _barrierResourceState.init(numberOfBarriers, maxProducersPerBarrier);
}

llvm::SmallVector<mlir::Operation*> BarrierScheduler::getConsumerOps(mlir::Operation* op) {
    return _taskConsumerMapOriginal[op];
}

mlir::IntegerAttr BarrierScheduler::getUniqueID(mlir::Operation* op) {
    auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
    return taskOp->getAttr(uniqueIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
}

void BarrierScheduler::assignTaskPriorities() {
    _log.trace("Assigning task priorities");
    _log = _log.nest();

    operationInDegreeType inDegree = _originalInDegree;

    // Assign topological sort level as priority
    std::list<mlir::Operation*> zeroInDegreeNodes[2];
    _priority.clear();

    size_t currentPriority = 0;

    auto itr = _inDegree.begin();
    while (itr != _inDegree.end()) {
        auto op = itr->first;
        auto opDegreeIt = _inDegree.find(op);
        if (opDegreeIt != _inDegree.end() && opDegreeIt->second == 0) {
            _log.trace("Adding task {0} to zeroInDegreeNodes ", getUniqueID(op));
            zeroInDegreeNodes[currentPriority % 2].push_back(op);
            _log.trace("The priority for  op {0}  is {1}", getUniqueID(op), currentPriority);
            _priority[op] = currentPriority;
        }
        ++itr;
    }

    while (!zeroInDegreeNodes[currentPriority % 2].empty()) {
        // decrement the in-degree
        for (auto op = zeroInDegreeNodes[currentPriority % 2].begin();
             op != zeroInDegreeNodes[currentPriority % 2].end(); ++op) {
            auto opConsumers = getConsumerOps(*op);

            SmallVector<mlir::Operation*>::iterator jtr = opConsumers.begin();
            while (jtr != opConsumers.end()) {
                _log.trace("Looking up task {0} in the inDegree table ", getUniqueID(*jtr));
                typename operationInDegreeType::iterator deg_itr = inDegree.find(*jtr);

                VPUX_THROW_UNLESS((deg_itr != inDegree.end()) && (deg_itr->second > 0), "Invalid indegree");

                (deg_itr->second)--;

                if (!(deg_itr->second)) {
                    // in-degree of this node has become zero//
                    _log.trace("The in-degree of op task {0}  has become zero ", getUniqueID(deg_itr->first));

                    _log.trace("The priority of task {0}  has become  {1} ", getUniqueID(deg_itr->first),
                               (currentPriority + 1));

                    _priority[deg_itr->first] = (currentPriority + 1);
                    zeroInDegreeNodes[(currentPriority + 1) % 2].push_back(deg_itr->first);

                    _log.trace("Erasing task {0} from the in-degree table ", getUniqueID(deg_itr->first));
                    inDegree.erase(deg_itr);
                }
                ++jtr;
            }
        }
        zeroInDegreeNodes[currentPriority % 2].clear();
        ++currentPriority;
    }

    for (typename priorityMapType::iterator pitr = _priority.begin(); pitr != _priority.end(); ++pitr) {
        _log.trace("Checking priority of {0} ", getUniqueID(pitr->first));
        auto opConsumers = getConsumerOps((pitr->first));

        // set priority to max of all out going priorities //
        SmallVector<mlir::Operation*>::iterator jtr = opConsumers.begin();

        if (!(pitr->second)) {
            size_t max = pitr->second;
            while (jtr != opConsumers.end()) {
                max = std::max(_priority[*jtr], max);
                ++jtr;
            }
            pitr->second = max;
        }
    }

    struct custom_compare final {
        bool operator()(const std::pair<size_t, mlir::Operation*>& left,
                        const std::pair<size_t, mlir::Operation*>& right) const {
            size_t priorityLeft = left.first;
            size_t priorityRight = right.first;
            auto opIDLeft = getUniqueID(left.second).getInt();
            auto opIDright = getUniqueID(right.second).getInt();

            if (priorityLeft < priorityRight)
                return true;

            if (priorityLeft > priorityRight)
                return false;

            return opIDLeft < opIDright;
        }
    };

    // reassign the priority
    std::set<std::pair<size_t, mlir::Operation*>, custom_compare> s;  // The new (temporary) container.
    for (auto const& pair : _priority)
        s.emplace(pair.second, pair.first);  // Flip the pairs.

    size_t newPriority = 1;
    for (auto const& pair : s) {
        _priority[pair.second] = newPriority++;
    }

    _log = _log.unnest();
}

void BarrierScheduler::assignTaskUniqueIds() {
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
size_t BarrierScheduler::countProducerTasksToBarrier(mlir::Operation* op) {
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::NCE) {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
        auto innerTaskOp = taskOp.getInnerTaskOp();
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerTaskOp)) {
            innerTaskOp = clusterTilingOp.getInnerTaskOp();
        }
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(innerTaskOp);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::DMA_NN ||
        mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA ||
        mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        return 1;
    } else {
        VPUX_THROW("This operation does not run on hardware");
    }
}

// This function creates a table storing the number of producers to a barrier that a task requires.
void BarrierScheduler::createTaskBarrierResourceUtilityTable() {
    _log = _log.nest();
    for (auto& task : _originalInDegree) {
        auto barrierResouceUtilization = countProducerTasksToBarrier(task.first);
        _log.trace("Task {0} requires {1} barrier producer slots", getUniqueID(task.first), barrierResouceUtilization);
        _barrierResourceUtilizationMap.insert(std::make_pair(task.first, barrierResouceUtilization));
    }
    _log = _log.unnest();
}

void BarrierScheduler::init() {
    _log.trace("Feasible barrier scheduler initialization");
    _log = _log.nest();

    // Assign unique IDs to tasks
    _log.trace("Assigning unique IDs to tasks");
    assignTaskUniqueIds();

    // Save the original IR dependency. The IR may need to be restored
    // if barrier simulation fails after the barrier scheduler has run.
    // If barrier simulation does fail, the IR is restored and another schedule
    // is generated
    _log.trace("Saving the original IR dependency");
    saveOriginalIRDependency();

    // Assign task priorities
    _log.trace("Assigning task scheduling priorities");
    assignTaskPriorities();

    // Store per-task barrier producer utilization, i.e. the number of workloads
    _log.trace("Collating per task, the barrier resource requirements");
    createTaskBarrierResourceUtilityTable();
    _log = _log.unnest();
}

bool BarrierScheduler::generateScheduleWithBarriers(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    bool scheduleSuccess = false;
    _processedTasks.clear();
    _schedulableCandidates.clear();
    _scheduledTasks.clear();
    _barrierAssociationTable.clear();
    _heap.clear();
    _barrierMap.clear();
    _barrierCount = numberOfBarriers;
    _slotsPerBarrier = maxProducersPerBarrier;
    _inDegree = _originalInDegree;
    _currentTime = 0;

    _log.trace("Starting to generate a schedule with {0} barriers", numberOfBarriers);
    _log = _log.nest();

    // retrieve output ops (ops with zero out-degree)
    _outputTasks = _originalOutputOps;

    // Create a barrier transition structure per barrier
    initializeBarrierAssociationTable();

    _log.trace("Initializing the barrier resource upper state i.e. maximum barrier and maximum producers per barrier");
    initializeBarrierResourceState(numberOfBarriers, maxProducersPerBarrier);

    auto itr = _inDegree.begin();
    while (itr != _inDegree.end()) {
        auto op = itr->first;
        auto opDegreeIt = _inDegree.find(op);
        if (opDegreeIt != _inDegree.end() && opDegreeIt->second == 0) {
            _log.nest().trace("Adding task: {0} to candidate set", getUniqueID(op));
            addTaskToCandidateSet(op);
        }
        ++itr;
    }

    VPUX_THROW_UNLESS(!_schedulableCandidates.empty(),
                      "No operations with zero in-degree exist, error processing the dependencies");

    // Scheduling loop, loop until all output tasks are scheduled
    scheduleSuccess = performSchedulingTaskLoop();
    VPUX_THROW_UNLESS(scheduleSuccess == true, "Failed to generate a valid schedule");

    // Insert barriers in the IR based on the output of the list scheduler
    _log.trace("Inserting barriers in the IR");
    insertBarriersinIR();

    _log = _log.unnest();

    _log.trace("Finished generating a schedule with barriers");

    return scheduleSuccess;
}

void BarrierScheduler::reorderIR() {
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

void BarrierScheduler::insertBarriersinIR() {
    size_t schedulingNumber = 0UL;
    size_t barrierCount = 0UL;
    mlir::OpBuilder builder(_func.getBody());

    _log = _log.nest();
    _log.trace("Processing the scheduled tasks");
    for (const auto& op : _scheduledTasks) {
        auto bitr = _barrierAssociationTable.find(op._barrierIndex);

        VPUX_THROW_UNLESS(bitr != _barrierAssociationTable.end(),
                          "Unable to find barrier index {0} in the barrier association table");

        barrierTransitionStructure& bstructure = bitr->second;

        // Set scheduling number
        _log.trace("Assigning scheduling number {0} to the task {1} ", schedulingNumber, getUniqueID(op._op));
        op._op->setAttr(schedulingNumberAttrName, getIntAttr(op._op->getContext(), schedulingNumber));
        _schedulingOrder.push_back(getUniqueID(op._op).getInt());

        schedulingNumber++;

        // STEP-2: update barrier structure invariant //
        bool newBarrierTaskCreated = bstructure.processNextScheduledTask(op, builder);

        if (newBarrierTaskCreated) {
            ++barrierCount;
        }
    }

    // STEP-2.5: process trailing barrier control structures //
    {
        for (auto bitr = _barrierAssociationTable.begin(); bitr != _barrierAssociationTable.end(); ++bitr) {
            barrierTransitionStructure& bstruct = bitr->second;
            bstruct.closeBarrierProducerList();
        }
    }

    _log.trace("Barrier scheduling complete");

    populateTasksUpdateWaitBarrierMap(_configureBarrierOpWaitMap, _configureBarrierOpUpdateMap, _configureTaskOpWaitMap,
                                      _configureTaskOpUpdateMap);

    _log.trace("Removing redundant dependencies");
    removeRedundantDependencies();

    _log.trace("Removing redundant barriers");
    removeRedundantBarriers();

    for (size_t ind = 0; ind < _configureBarrierOpUpdateMap.size(); ind++) {
        _log.trace("Virtual Barrier ID {0} has {1} consumers", ind, _configureBarrierOpUpdateMap[ind].count());
    }

    for (size_t ind = 0; ind < _configureBarrierOpWaitMap.size(); ind++) {
        _log.trace("Virtual Barrier ID {0} has {1} producers", ind, _configureBarrierOpWaitMap[ind].count());
    }

    _log.trace("Starting to add barriers into the IR");
    for (size_t ind = 0; ind < _configureBarrierOpWaitMap.size(); ind++) {
        auto& barrierOp = _orderedBarrier[ind];
        auto waitMap = _configureBarrierOpWaitMap[ind].set_bits();
        auto updateMap = _configureBarrierOpUpdateMap[ind].set_bits();
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
}

void BarrierScheduler::removeVirtualBarriers() {
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

void BarrierScheduler::clearTemporaryAttributes() {
    _func->walk([](VPURT::TaskOp op) {
        op->removeAttr(uniqueIdAttrName);
        op->removeAttr(virtualIdAttrName);
        op->removeAttr(schedulingNumberAttrName);
    });
}

bool BarrierScheduler::performRuntimeSimulation() {
    bool success = true;

    _log.trace("Starting runtime simulation");
    reorderIR();
    if (_orderedBarrier.size()) {
        // run simulation
        VPURT::BarrierSimulator barrierSim(_func);
        VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");
        success = mlir::succeeded(barrierSim.simulateBarriers(_log.nest()));
    }

    if (!success) {
        _log.trace("Barrier simulation was not successful removing the barriers that were inserted");
        removeVirtualBarriers();
        _configureBarrierOpWaitMap.clear();
        _configureBarrierOpUpdateMap.clear();
        _configureTaskOpWaitMap.clear();
        _configureTaskOpUpdateMap.clear();
        _orderedBarrier.clear();
        _schedulingOrder.clear();
    }

    _log.trace("Barrier simulation result is {0} with upperbound {1}", success, _barrierCount);
    return success;
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
void BarrierScheduler::removeRedundantBarriers() {
    for (size_t ind = 0; ind < _configureBarrierOpUpdateMap.size(); ind++) {
        auto& consumers = _configureBarrierOpUpdateMap[ind];
        if (!(consumers.set_bits().empty())) {
            auto ind1 = ind + 1;
            for (; ind1 < _configureBarrierOpUpdateMap.size(); ind1++) {
                auto& consumers1 = _configureBarrierOpUpdateMap[ind1];
                if (consumers1 == consumers) {
                    _log.trace("Found barrier {0} and {1} have same consumers", ind, ind1);
                    auto& producers = _configureBarrierOpWaitMap[ind1];
                    size_t variantsCount = 0;
                    size_t invariantsCount = 0;
                    for (auto oldProducer : _configureBarrierOpWaitMap[ind].set_bits()) {
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
                    size_t invariantsLimits = (numClusters == 1) ? 32 : 64;
                    if ((variantsCount <= _slotsPerBarrier) && (invariantsCount <= invariantsLimits)) {
                        for (auto task : producers.set_bits()) {
                            _configureBarrierOpWaitMap[ind].set(task);
                        }
                        producers.reset();
                        consumers1.reset();
                    }
                }
            }
        }
    }

    for (size_t ind = 0; ind < _configureBarrierOpWaitMap.size(); ind++) {
        auto& producers = _configureBarrierOpWaitMap[ind];
        auto& consumers = _configureBarrierOpUpdateMap[ind];
        if (!(producers.set_bits().empty() || consumers.set_bits().empty())) {
            auto prod = producers.set_bits_begin();
            bool producersOnlyHasDMA = true;
            for (; prod != producers.set_bits_end(); prod++) {
                if (_orderedTasks[*prod].getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
                    producersOnlyHasDMA = false;
                    break;
                }
            }

            if (producersOnlyHasDMA) {
                auto cons = consumers.set_bits_begin();
                bool consumersOnlyHasDMA = true;
                for (; cons != consumers.set_bits_end(); cons++) {
                    if (_orderedTasks[*cons].getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
                        consumersOnlyHasDMA = false;
                        break;
                    }
                }

                if (consumersOnlyHasDMA) {
                    producers.reset();
                    consumers.reset();
                }
            }
        }
    }
}

// For two producers {a, b} of a barrier, if a depends on b then b isn't a necessary producer for this barrier
// For two consumers {a, b} of a barrier, if a depends on b then a isn't a necessary consumer for this barrier
void BarrierScheduler::removeRedundantDependencies() {
    for (size_t ind = 0; ind < _configureBarrierOpWaitMap.size(); ind++) {
        // producers
        auto& producers = _configureBarrierOpWaitMap[ind];
        SmallVector<unsigned> producersToRemove;
        for (auto prod = producers.set_bits_begin(); prod != producers.set_bits_end(); prod++) {
            auto prod1 = prod;
            prod1++;
            for (; prod1 != producers.set_bits_end(); prod1++) {
                if (doesPathExist(*prod1, *prod)) {
                    producersToRemove.push_back(*prod1);
                } else if (doesPathExist(*prod, *prod1)) {
                    producersToRemove.push_back(*prod);
                    break;
                }
            }
        }

        for (auto& producer : producersToRemove) {
            producers.reset(producer);
        }

        // consumers
        auto& consumers = _configureBarrierOpUpdateMap[ind];
        SmallVector<unsigned> consumersToRemove;
        for (auto cons = consumers.set_bits_begin(); cons != consumers.set_bits_end(); cons++) {
            auto cons1 = cons;
            cons1++;
            for (; cons1 != consumers.set_bits_end(); cons1++) {
                if (doesPathExist(*cons, *cons1)) {
                    consumersToRemove.push_back(*cons1);
                } else if (doesPathExist(*cons1, *cons)) {
                    consumersToRemove.push_back(*cons);
                    break;
                }
            }
        }

        for (auto& consumer : consumersToRemove) {
            consumers.reset(consumer);
        }
    }
}

void BarrierScheduler::initializeBarrierAssociationTable() {
    _log.trace("Step-0: Initialize the barrier association table");
    for (size_t barrierId = 0; barrierId < _barrierCount; barrierId++) {
        auto bitr = _barrierAssociationTable.insert(
                std::make_pair(barrierId, barrierTransitionStructure(*this, _taskCount)));
        barrierTransitionStructure& bstructure = (bitr.first)->second;
        bstructure.init();
    }
}

// detect if op b depends on a
bool BarrierScheduler::doesPathExist(int64_t a, int64_t b) {
    auto numa = _orderedTasks[a]->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
    auto numb = _orderedTasks[b]->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
    if (numa >= numb)
        return false;
    else {
        // DMAs which are scheduled later in the schedule naturally depend on DMAs which were scheduled previously
        if ((_orderedTasks[a].getExecutorKind() == VPU::ExecutorKind::DMA_NN) &&
            (_orderedTasks[b].getExecutorKind() == VPU::ExecutorKind::DMA_NN)) {
            return true;
        }
        auto updateBarriers = _configureTaskOpUpdateMap[a];
        for (auto updateBarrier : updateBarriers.set_bits()) {
            auto barrierConsumers = _configureBarrierOpUpdateMap[updateBarrier];
            for (auto consumer : barrierConsumers.set_bits()) {
                if (consumer == b)
                    return true;
                if (doesPathExist(consumer, b))
                    return true;
            }
        }
        return false;
    }
}

void BarrierScheduler::populateScheduledTasks(mlir::Operation* task) {
    _log.trace("Populating the scheduling info for the scheduled task {0}", getUniqueID(task));
    _log = _log.nest();
    ScheduledOpInfo scheduledTask;

    scheduledTask._op = task;
    scheduledTask._scheduleTime = _currentTime;

    _log.trace("Get barrier info for task{0}", getUniqueID(task));
    const barrierInfo& binfo = getBarrierInfo(task);

    scheduledTask._barrierIndex = binfo._barrierIndex;
    scheduledTask._producerSlotCount = binfo._producerSlotCount;

    _log.trace("Task {0} is scheduled in time  {1}", getUniqueID(task), scheduledTask._scheduleTime);
    _log.trace("The task's barrier index is {0} and the slot count is {1}", scheduledTask._barrierIndex,
               scheduledTask._producerSlotCount);

    _scheduledTasks.push_back(scheduledTask);
    _log = _log.unnest();
}
