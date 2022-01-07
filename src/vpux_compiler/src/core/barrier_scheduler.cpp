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

#include "vpux/compiler/core/barrier_scheduler.hpp"

using namespace vpux::VPURT;

//
// Barrier Scheduler
//
// Barriers are physical synchronization primitives in VPU hardware. Every task 
// requires a barrier resource to indicate its completion to its adjacent tasks. 
// Thus every scheduled operation with at least one outgoing edge needs an update barrier.
//
// In the feasible memory scheduler, a feasible schedule is generated. However, VPU hardware 
// only has a finite number of barriers, 8 per cluster. The Barrier scheduler class ensures 
// that in all possible schedules of the number of active barriers does not exceed the available
// physical barriers for the target platform. To achieve this the scheduler may need to insert 
// some additional control dependencies among the tasks. 

// The hardware allows a finite producer count of 256 for each of the update barriers.
// This means that multiple tasks can update the same active barrier. This is incorporated into the
// barrier resource model by using the term barrier slots.
// In addition to the upper bound of available barriers it is assumed that each of these barriers has 
// a maximum of 256 slots. The barrier demand is expressed as the number of slots required. 
// In the context of VPU hardware the number of slots for a DPU tasks are the DPU worklaods 
// and for a DMA tasks it is 1. 

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
          _log(log),
          _func(func){};

void BarrierScheduler::populateTasksUpdateWaitBarrierMap(barrierUpdateWaitMapType& barrierOpUpdateWaitMap,
                                                                 taskOpUpdateWaitMapType& taskOpUpdateWaitMap) {
    for (auto iter = barrierOpUpdateWaitMap.begin(); iter != barrierOpUpdateWaitMap.end(); iter++) {
        auto barrierOp = (*iter).first;
        auto producers = (*iter).second.first;
        auto consumers = (*iter).second.second;
        for (auto prod = producers.begin(); prod != producers.end(); prod++) {
            auto taskUpateItr = taskOpUpdateWaitMap.find(*prod);
            if (taskUpateItr != taskOpUpdateWaitMap.end()) {
                taskUpateItr->second.second.insert(barrierOp);
            } else {
                std::set<mlir::Operation*> newBarrierProducers{};
                std::set<mlir::Operation*> newBarrierConsumers{barrierOp};
                taskOpUpdateWaitMap.insert(
                        std::make_pair(*prod, std::make_pair(newBarrierProducers, newBarrierConsumers)));
            }
        }

        for (auto cons = consumers.begin(); cons != consumers.end(); cons++) {
            auto taskWaitItr = taskOpUpdateWaitMap.find(*cons);
            if (taskWaitItr != taskOpUpdateWaitMap.end()) {
                taskWaitItr->second.first.insert(barrierOp);
            } else {
                std::set<mlir::Operation*> newBarrierProducers{barrierOp};
                std::set<mlir::Operation*> newBarrierConsumers{};
                taskOpUpdateWaitMap.insert(
                        std::make_pair(*cons, std::make_pair(newBarrierProducers, newBarrierConsumers)));
            }
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
        _taskConsumerMapBackUp.insert(std::make_pair(taskOp, consumers));

        if (consumers.empty())
            _origninalOutputOps.insert(taskOp);
    };

    _func->walk([&](VPURT::TaskOp taskOp) {
        switch (taskOp.getExecutorKind()) {
        case VPU::ExecutorKind::DMA_NN: {
            updateBarrierConfigs(taskOp);
            break;
        }
        case VPU::ExecutorKind::NCE: {
            updateBarrierConfigs(taskOp);
            break;
        }
        case VPU::ExecutorKind::SHAVE_ACT: {
            updateBarrierConfigs(taskOp);
            break;
        }
        case VPU::ExecutorKind::SHAVE_UPA: {
            updateBarrierConfigs(taskOp);
            break;
        }
        default:
            VPUX_THROW("Unsupported executor '{0}'", taskOp.getExecutorKind());
        }
    });

    _func->walk([&](VPURT::TaskOp taskOp) {
        switch (taskOp.getExecutorKind()) {
        case VPU::ExecutorKind::DMA_NN: {
            updateInDegreeAndConsumers(taskOp);
            break;
        }
        case VPU::ExecutorKind::NCE: {
            updateInDegreeAndConsumers(taskOp);
            break;
        }
        case VPU::ExecutorKind::SHAVE_ACT: {
            updateInDegreeAndConsumers(taskOp);
            break;
        }
        case VPU::ExecutorKind::SHAVE_UPA: {
            updateInDegreeAndConsumers(taskOp);
            break;
        }
        default:
            VPUX_THROW("Unsupported executor '{0}'", taskOp.getExecutorKind());
        }
    });

    // Remove the original virtual barriers, optimal barriers are inserted based on the generated schedule
    removeVirtualBarriers();

    _log.trace("Removed all the original declare virtual barrier ops");
}

void BarrierScheduler::pushToHeap(const HeapElement& elem) {
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

    _log.trace("Adding operation  to candidates list {0} to candidates list", getUniqueID(op));
    _schedulableCandidates.push_back(op);
    _processedTasks.insert(op);
}

void BarrierScheduler::addOutGoingOperationsToCandidateList(mlir::Operation* op) {
    _log.trace("Add outgoing operations to candidate list");

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
}

bool BarrierScheduler::performSchedulingTaskLoop() {
    // Scheduling loop, loop until all output tasks are scheduled
    while (!_outputTasks.empty()) {
        schedulableTasksIteratorType taskItr = findSchedulableTask();

        if (isTasKInSchedulableCandidates(taskItr)) {
            // Found a schedulable task
            mlir::Operation* task = (*taskItr);
            _log.trace("Found a schedulable task ID {0}", getUniqueID(task));

            const size_t opDelay = 1;
            size_t taskBarrierResourceRequirement = _barrierResourceUtilizationMap[task];
            size_t taskEndTime = _currentTime + opDelay;

            _log.trace("Task ID {0} end time is {1}, pushing to heap", getUniqueID(task), taskEndTime);
            pushToHeap(HeapElement(task, taskEndTime));

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
                    "An error has occured the _currentScheduleTime should not be less than time poped from the heap");

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

    return true;
}

bool BarrierScheduler::isTasKInSchedulableCandidates(schedulableTasksIteratorType itr) const {
    return !(itr == _schedulableCandidates.end());
}

BarrierScheduler::schedulableTasksIteratorType BarrierScheduler::findSchedulableTask() {
    _log.trace("Looking for a a scheduleable task");

    schedulableTasksIteratorType itr = _schedulableCandidates.end();
    std::list<schedulableTasksIteratorType> readyList;

    _log.trace("There are {0} candiates", _schedulableCandidates.size());

    for (itr = _schedulableCandidates.begin(); itr != _schedulableCandidates.end(); ++itr) {
        _log.trace("The producerSlotRequirement for task {0} is {1}", getUniqueID(*itr),
                   _barrierResourceUtilizationMap[*itr]);

        if (isBarrierResourceAvailable(_barrierResourceUtilizationMap[*itr])) {
            _log.trace("Adding task {0} to the ready list", getUniqueID(*itr));
            readyList.push_back(itr);
        }
    }

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

size_t BarrierScheduler::currentTime() const {
    return _currentTime;
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
    auto unassignBarrierSlots = _barrierResourceState.unassignBarrierSlots(binfo._barrierIndex, binfo._producerSlotCount);

    VPUX_THROW_UNLESS(unassignBarrierSlots == true, "Failed to dealocate slots in the barrier index {0}", binfo._barrierIndex);
    _barrierMap.erase(itr);
    return true;
}

bool BarrierScheduler::scheduleTask(mlir::Operation* op, const size_t producerSlotRequirement) {
    _log.trace("Scheduling a task");

    VPUX_THROW_UNLESS(isBarrierResourceAvailable(producerSlotRequirement) == true,
                      "Attempt to schedule task failed, failed to allocate barrier resource for task {0}}",
                      getUniqueID(op));

    if (_barrierMap.find(op) != _barrierMap.end()) {
        return false;
    }
    size_t barrierID = _barrierResourceState.assignBarrierSlots(producerSlotRequirement);
    _barrierMap.insert(std::make_pair(op, barrierInfo(barrierID, producerSlotRequirement)));
    return true;
}

bool BarrierScheduler::isBarrierResourceAvailable(const size_t producerSlotRequirement) {
    return _barrierResourceState.hasBarrierWithSlots(producerSlotRequirement);
}

void BarrierScheduler::initializeBarrierResourceState(const size_t numberOfBarriers,
                                                              const size_t maxProducersPerBarrier) {
    _barrierResourceState.init(numberOfBarriers, maxProducersPerBarrier);
}

llvm::SmallVector<mlir::Operation*> BarrierScheduler::getConsumerOps(mlir::Operation* op) {
    return _taskConsumerMapBackUp[op];
}

mlir::IntegerAttr BarrierScheduler::getUniqueID(mlir::Operation* op) {
    auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
    return taskOp->getAttr(uniqueIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
}

void BarrierScheduler::assignTaskPriorities() {
    operationInDegreeType inDegree = _originalInDegree;

    // Assign topological sort level as priority
    std::list<mlir::Operation*> zeroInDegreeNodes[2];
    _priority.clear();

    size_t currentPriority = 0;

    operationInDegreeType::iterator itr = _inDegree.begin();
    while (itr != _inDegree.end()) {
        auto op = itr->first;
        if (_inDegree.find(op)->second == 0) {
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
}

void BarrierScheduler::assignTaskUniqueIds() {
    int64_t uniqueId = 0;
    auto assignUniqueIDs = [&](VPURT::TaskOp taskOp) {
        taskOp->setAttr(uniqueIdAttrName, getIntAttr(taskOp->getContext(), uniqueId++));
    };

    _func.walk([&](VPURT::TaskOp taskOp) {
        switch (taskOp.getExecutorKind()) {
        case VPU::ExecutorKind::DMA_NN: {
            assignUniqueIDs(taskOp);
            break;
        }
        case VPU::ExecutorKind::NCE: {
            assignUniqueIDs(taskOp);
            break;
        }
        case VPU::ExecutorKind::SHAVE_UPA: {
            assignUniqueIDs(taskOp);
            break;
        }
        case VPU::ExecutorKind::SHAVE_ACT: {
            assignUniqueIDs(taskOp);
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getExecutorKind());
        }
    });
}

bool BarrierScheduler::doesOpRunOnNCE(mlir::Operation* op) {
    if ((mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::NCE) ||
        (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::DMA_NN)) {
        return true;
    } else {
        return false;
    }
}

size_t BarrierScheduler::countProducerConsumerTasks(mlir::Operation* op) {
    const size_t dmaBarrierProducerUtilization = 1;
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::NCE) {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
        auto& block = taskOp.body().getBlocks().front();
        auto wrappedTaskOp = block.begin();
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
        return dmaBarrierProducerUtilization;
    } else {
        VPUX_THROW("This operation does not run on hardware");
    }
}

void BarrierScheduler::createTaskBarrierResourceUtilityTable() {
    const size_t upaTaskbarrierResouceUtilization = 0;
    for (auto& task : _originalInDegree) {
        if (doesOpRunOnNCE(task.first)) {
            auto barrierResouceUtilization = countProducerConsumerTasks(task.first);
            _log.trace("Task {0} is a DPU or DMA task and requires {1} barrier producer slots", getUniqueID(task.first),
                       barrierResouceUtilization);
            _barrierResourceUtilizationMap.insert(std::make_pair(task.first, barrierResouceUtilization));
        } else  // UPA tasks
        {
            _log.trace("Task: {0} is a UPA tasks and requires 0 barrier producer slots", getUniqueID(task.first));
            _barrierResourceUtilizationMap.insert(std::make_pair(task.first, upaTaskbarrierResouceUtilization));
        }
    }
}

void BarrierScheduler::init() {
    _log.trace("Feasible barrier scheduler initialization");

    // Assing unique IDs to tasks
    _log.trace("Assigning unique IDs to tasks");
    assignTaskUniqueIds();

    // Save the original IR dependency. The IR may need to be restored
    // if barrier simulation fails after the barrier scheduler has run.
    // If barrier simulation does fail, the IR is restored and another schedule
    // is generated
    _log.trace("Saving the original IR dependency");
    saveOriginalIRDependency();

    // Assign task priorities
    _log.trace("Assiging task scheduling priorities");
    assignTaskPriorities();

    // Store per-task barrier producer utilization, i.e. the number of workloads
    _log.trace("Collating per task, the barrier resource requirements");
    createTaskBarrierResourceUtilityTable();
}

bool BarrierScheduler::generateScheduleWithBarriers(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    bool scheduleSuccess = false;
    _processedTasks.clear();
    _schedulableCandidates.clear();
    _scheduledTasks.clear();
    _barrierAssociationTable.clear();
    _barrierCount = numberOfBarriers;
    _slotsPerBarrier = maxProducersPerBarrier;
    _inDegree = _originalInDegree;

    // retrieve output ops (ops with zero out-degree)
    _outputTasks = _origninalOutputOps;

    // Create a barrier transition structure per barrier
    initializeBarrierAssociationTable();

    _log.trace("Initializing the barrier resource upper state i.e. maximum barrier and maximum producers per barrier");
    initializeBarrierResourceState(numberOfBarriers, maxProducersPerBarrier);

    operationInDegreeType::iterator itr = _inDegree.begin();
    while (itr != _inDegree.end()) {
        auto op = itr->first;
        if (_inDegree.find(op)->second == 0) {
            _log.trace("Adding task: {0} to candidate set", getUniqueID(op));
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

    return scheduleSuccess;
}

void BarrierScheduler::reorderIR() {
    // reorder barrier by id
    mlir::Operation* preBarrier = nullptr;
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        auto curBarrier = (*iter).first;
        if (preBarrier) {
            curBarrier->moveAfter(preBarrier);
        }
        preBarrier = curBarrier;
    }

    // reorder task by scheduling number
    mlir::Operation* preTask = nullptr;
    for (auto iter = configureTaskOpUpdateWaitMap.begin(); iter != configureTaskOpUpdateWaitMap.end(); iter++) {
        auto curTask = (*iter).first;
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

    for (const auto& op : _scheduledTasks) {
        auto bitr = _barrierAssociationTable.find(op._barrierIndex);

        VPUX_THROW_UNLESS(bitr != _barrierAssociationTable.end(),
                          "Unable to find barrier index {0} in the barrier association table");

        barrierTransitionStructure& bstructure = bitr->second;

        // Set scheduling number
        _log.trace("Assigning scheduling number {0} to the task {1} ", schedulingNumber, getUniqueID(op._op));
        op._op->setAttr(schedulingNumberAttrName, getIntAttr(op._op->getContext(), schedulingNumber));

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

    populateTasksUpdateWaitBarrierMap(configureBarrierOpUpdateWaitMap, configureTaskOpUpdateWaitMap);

    removeRedundantDependencies();
    removeRedundantBarriers();

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        _log.trace("Virtual Barrier ID {0} has {1} consumers", barrierOp->getAttr("id"), p.second.second.size());
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        _log.trace("Virtual Barrier ID {0} has {1} producers", barrierOp->getAttr("id"), p.second.first.size());
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second.first) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);

            VPUX_THROW_UNLESS(taskOp != NULL, "Invalid task");
            VPUX_THROW_UNLESS(barrierOp.barrier() != NULL, "Invalid barrier");
            _log.trace("Adding Barrier ID {0} as an update barrier for operation {1}", barrierOp->getAttr("id"),
                       getUniqueID(user));
            taskOp.updateBarriersMutable().append(barrierOp.barrier());
        }
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second.second) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);

            VPUX_THROW_UNLESS(taskOp != NULL, "Invalid task");
            VPUX_THROW_UNLESS(barrierOp.barrier() != NULL, "Invalid barrier");
            _log.trace("Adding Barrier ID {0} as an wait barrier for operation {1}", barrierOp->getAttr("id"),
                       getUniqueID(user));
            taskOp.waitBarriersMutable().append(barrierOp.barrier());
        }
    }
}

void BarrierScheduler::removeVirtualBarriers() {
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
    });
}

bool BarrierScheduler::performRuntimeSimulation() {
    bool success = true;
    reorderIR();
    if (configureBarrierOpUpdateWaitMap.size()) {
        // run simulation
        VPURT::BarrierSimulator barrierSim(_func);
        VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");
        success = mlir::succeeded(barrierSim.simulateBarriers(_log.nest()));
    }

    if (!success) {
        removeVirtualBarriers();
        configureBarrierOpUpdateWaitMap.clear();
        configureTaskOpUpdateWaitMap.clear();
    }

    _log.trace("Barrier simualtion result is {0} with upperbound {1}", success, _barrierCount);
    return success;
}

// If two barriers have same consumers, they can be merged
// If a barrier has no producers, it can be removed
void BarrierScheduler::removeRedundantBarriers() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        auto consumers = (*iter).second.second;
        if (!consumers.empty()) {
            auto iter1 = iter;
            iter1++;
            for (; iter1 != configureBarrierOpUpdateWaitMap.end(); iter1++) {
                auto& consumers1 = (*iter1).second.second;
                if (consumers1 == consumers) {
                    _log.trace("found barrier {0} and {1} have same consumers", (*iter).first->getAttr("id"),
                               (*iter1).first->getAttr("id"));
                    auto& producers = (*iter1).second.first;
                    for (auto task : producers) {
                        (*iter).second.first.insert(task);
                    }
                    producers.clear();
                    consumers1.clear();
                }
            }
        }
    }

    for (auto itr = configureBarrierOpUpdateWaitMap.begin(); itr != configureBarrierOpUpdateWaitMap.end();) {
        if (itr->second.first.empty() || itr->second.second.empty()) {
            _log.trace("Earsing virtual Barrier ID {0} as it has no producers", itr->first->getAttr("id"));
            (*itr).first->dropAllUses();
            (*itr).first->erase();
            itr = configureBarrierOpUpdateWaitMap.erase(itr);
        } else {
            ++itr;
        }
    }
}

// For two producers {a, b} of a barrier, if a depends on b then b isn't a necessary producer for this barrier
// For two consumers {a, b} of a barrier, if a depends on b then a isn't a necessary consumer for this barrier
void BarrierScheduler::removeRedundantDependencies() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        // producers
        auto& producersSet = (*iter).second.first;
        SmallVector<mlir::Operation*> producers(producersSet.begin(), producersSet.end());
        SmallVector<mlir::Operation*> producersToRemove;
        for (auto prod = producers.begin(); prod != producers.end(); prod++) {
            auto prod1 = prod;
            prod1++;
            for (; prod1 != producers.end(); prod1++) {
                if (doesPathExist(*prod1, *prod)) {
                    producersToRemove.push_back(*prod1);
                    *prod1 = NULL;
                } else if (doesPathExist(*prod, *prod1)) {
                    producersToRemove.push_back(*prod);
                    *prod = NULL;
                    break;
                }
            }
        }

        for (auto& producer : producersToRemove) {
            producersSet.erase(producer);
        }

        // consumers
        auto& consumersSet = (*iter).second.second;
        SmallVector<mlir::Operation*> consumers(consumersSet.begin(), consumersSet.end());
        SmallVector<mlir::Operation*> consumersToRemove;
        for (auto cons = consumers.begin(); cons != consumers.end(); cons++) {
            auto cons1 = cons;
            cons1++;
            for (; cons1 != consumers.end(); cons1++) {
                if (doesPathExist(*cons, *cons1)) {
                    consumersToRemove.push_back(*cons1);
                    *cons1 = NULL;
                } else if (doesPathExist(*cons1, *cons)) {
                    consumersToRemove.push_back(*cons);
                    *cons = NULL;
                    break;
                }
            }
        }

        for (auto& consumer : consumersToRemove) {
            consumersSet.erase(consumer);
        }
    }
}

void BarrierScheduler::initializeBarrierAssociationTable() {
    _log.trace("STEP-0: Initialize the association table");
    for (size_t barrierId = 1; barrierId <= _barrierCount; barrierId++) {
        auto bitr = _barrierAssociationTable.insert(std::make_pair(barrierId, barrierTransitionStructure(*this)));
        barrierTransitionStructure& bstructure = (bitr.first)->second;
        bstructure.init();
    }
}

// detect if op b depends on a
bool BarrierScheduler::doesPathExist(mlir::Operation* a, mlir::Operation* b) {
    if (a == NULL || b == NULL)
        return false;
    auto numa = a->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
    auto numb = b->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
    if (numa >= numb)
        return false;
    else {
        auto updateBarriers = configureTaskOpUpdateWaitMap[a].second;
        std::set<mlir::Operation*> consumers;
        for (auto iter = updateBarriers.begin(); iter != updateBarriers.end(); iter++) {
            auto barrierConsumers = configureBarrierOpUpdateWaitMap[*iter].second;
            consumers.insert(barrierConsumers.begin(), barrierConsumers.end());
        }

        if (std::find(consumers.begin(), consumers.end(), b) != consumers.end())
            return true;
        else {
            for (auto consumer = consumers.begin(); consumer != consumers.end(); consumer++) {
                if (doesPathExist(*consumer, b))
                    return true;
            }
        }
        return false;
    }
}

void BarrierScheduler::populateScheduledTasks(mlir::Operation* task) {
    _log.trace("Populating the scheduling info for the scheduled task {0}", getUniqueID(task));
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
}