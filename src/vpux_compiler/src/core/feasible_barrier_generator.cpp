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

#include "vpux/compiler/core/feasible_barrier_generator.hpp"

using namespace vpux::VPURT;

FeasibleBarrierScheduler::FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log)
        : _barrierCount(),
          _slotsPerBarrier(),
          _resourceState(),
          _inDegree(),
          _heap(),
          _currentTime(0),
          _scheduleableCandidates(),
          _schedulableTask(),
          _processedTasks(),
          _priority(),
          _log(log),
          _ctx(ctx),
          _func(func){};

void FeasibleBarrierScheduler::getTaskOpUpdateWaitMap(
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                 operation_comparator_t>& barrierOpUpdateWaitMap,
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                 task_operation_comparator_by_schedule_time_t>& taskOpUpdateWaitMap) {
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

void FeasibleBarrierScheduler::saveOriginalIRDependency() {
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
        _inDegreeBackUp.insert(std::make_pair(taskOp, count));

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
            _outputOpsBackUp.insert(taskOp);
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

    cleanUpVirtualBarriers();

    _log.trace("Removed all the original declare virtual barrier ops");
}

void FeasibleBarrierScheduler::pushToHeap(const HeapElement& elem) {
    _heap.push_back(elem);
    std::push_heap(_heap.begin(), _heap.end(), MinHeapOrdering());
}

FeasibleBarrierScheduler::HeapElement FeasibleBarrierScheduler::popFromHeap() {
    std::pop_heap(_heap.begin(), _heap.end(), MinHeapOrdering());
    HeapElement elem = _heap.back();
    _heap.pop_back();
    return elem;
}

void FeasibleBarrierScheduler::addTaskToCandidateSet(mlir::Operation* op) {
    if (_processedTasks.find(op) != _processedTasks.end()) {
        return;
    }
    _log.trace("Adding operation  to candidates list {0} to candidates list", getUniqueID(op));
    _scheduleableCandidates.push_back(op);
    _processedTasks.insert(op);
}

void FeasibleBarrierScheduler::addOutGoingOperationsToCandidateList(mlir::Operation* op) {
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

        typename operation_in_degree_t::iterator deg_itr = _inDegree.find(op);

        VPUX_THROW_UNLESS((deg_itr != _inDegree.end()) && (deg_itr->second > 0), "Invalid indegree");
        assert((deg_itr != _inDegree.end()) && (deg_itr->second > 0));

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

bool FeasibleBarrierScheduler::scheduleOperations() {
    _schedulableTask = NULL;

    // scheduling loop, loop until all output ops are scheduled
    while (!_outputOps.empty()) {
        schedulable_ops_iterator_t op_itr = findSchedulableOp();

        if (isValidOp(op_itr)) {
            // found a schedulable operation //
            mlir::Operation* op = (*op_itr);

            delay_t op_delay = 1;
            resource_t op_resources = _resourceUtilityMap[*op_itr];
            schedule_time_t op_end_time = _currentTime + op_delay;

            _log.trace("Operation {0} end time is {1} pushing to heap", getUniqueID(*op_itr), op_end_time);
            pushToHeap(HeapElement(op, op_end_time));

            _scheduleableCandidates.erase(op_itr);

            // schedule operation
            scheduleOperation(op, op_resources);

            _schedulableTask = op;
            populateScheduledOps(op);
            _log.trace("The _schedulableTask ID is {0}", getUniqueID(_schedulableTask));
            // decrease outputs ops if output op scheduled
            if (_outputOps.find(op) != _outputOps.end()) {
                _outputOps.erase(op);
            }

        } else if (!_heap.empty()) {
            // no-op found so move up the schedule time to the smallest completion
            // time among the active operations. //
            HeapElement top_elem = popFromHeap();
            mlir::Operation* op = top_elem.op_;

            // assert(_currentTime <= top_elem.time_);
            VPUX_THROW_UNLESS(_currentTime <= top_elem.time_, "Invalid indegree");
            _currentTime = top_elem.time_;
            // since operation is now complete update the schedule //

            unScheduleOperation(op);
            // since op has completed add all out-going ops to candidates //
            addOutGoingOperationsToCandidateList(op);
        } else {
            // schedule is not feasible //
            _scheduleableCandidates.clear();
            break;
        }
    }

    // return _schedulableTask != NULL;
    return true;
}

bool FeasibleBarrierScheduler::isValidOp(schedulable_ops_iterator_t itr) const {
    return !(itr == _scheduleableCandidates.end());
}

FeasibleBarrierScheduler::schedulable_ops_iterator_t FeasibleBarrierScheduler::findSchedulableOp() {
    _log.trace("Looking for a a scheduleable operation");

    schedulable_ops_iterator_t itr = _scheduleableCandidates.end();
    std::list<schedulable_ops_iterator_t> ready_list;

    _log.trace("There are {0} candiates and for each candiate", _scheduleableCandidates.size());

    for (itr = _scheduleableCandidates.begin(); itr != _scheduleableCandidates.end(); ++itr) {
        _log.trace("The demand for operation {0} is {1}", getUniqueID(*itr), _resourceUtilityMap[*itr]);

        if (isResourceAvailable(_resourceUtilityMap[*itr])) {
            _log.trace("Adding operation {0} to the ready list", getUniqueID(*itr));
            ready_list.push_back(itr);
        }
    }

    _log.trace("Finding the operation with lowest priority in ready list");
    // find the one with lowest priority //
    if (!ready_list.empty()) {
        size_t min_priority = std::numeric_limits<size_t>::max();
        for (auto ritr = ready_list.begin(); ritr != ready_list.end(); ++ritr) {
            size_t currentPriority = _priority[*(*ritr)];
            if (currentPriority < min_priority) {
                itr = *ritr;
                min_priority = currentPriority;
            }
        }
    }
    return itr;
}

size_t FeasibleBarrierScheduler::currentTime() const {
    return _currentTime;
}

const resource_state_t& FeasibleBarrierScheduler::resourceState() const {
    return _resourceState;
}

bool FeasibleBarrierScheduler::unScheduleOperation(mlir::Operation*& op) {
    return _resourceState.unschedule_operation(op);
}

bool FeasibleBarrierScheduler::scheduleOperation(mlir::Operation*& op, resource_t demand) {
    return _resourceState.schedule_operation(op, demand);
}

bool FeasibleBarrierScheduler::isResourceAvailable(const resource_t& demand) {
    return _resourceState.is_resource_available(demand);
}

// TODO John improve this
void FeasibleBarrierScheduler::initializeBarrierResourceState(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    op_resource_state_t resource(numberOfBarriers, maxProducersPerBarrier);
    _resourceState.init(resource);
}

llvm::SmallVector<mlir::Operation*> FeasibleBarrierScheduler::getConsumerOps(mlir::Operation* op) {
    return _taskConsumerMapBackUp[op];
}

std::string FeasibleBarrierScheduler::printOpType(VPURT::TaskOp task) {
    if (task.getExecutorKind() == VPU::ExecutorKind::NCE)
        return "NCE task";
    if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN)
        return "DMA task ";
    if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA)
        return "Upa task ";

    return "task";
}

mlir::IntegerAttr FeasibleBarrierScheduler::getUniqueID(mlir::Operation* op) {
    auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
    return taskOp->getAttr(uniqueIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
}

void FeasibleBarrierScheduler::computeTaskPriorities() {
    operation_in_degree_t inDegree = _inDegreeBackUp;

    // Assign topological sort level as priority
    std::list<mlir::Operation*> zeroInDegreeNodes[2];
    _priority.clear();

    size_t currentPriority = 0;

    operation_in_degree_t::iterator itr = _inDegree.begin();
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
                typename operation_in_degree_t::iterator deg_itr = inDegree.find(*jtr);

                VPUX_THROW_UNLESS((deg_itr != inDegree.end()) && (deg_itr->second > 0), "Invalid indegree");

                assert((deg_itr != inDegree.end()) && (deg_itr->second > 0));
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

    for (typename priority_map_t::iterator pitr = _priority.begin(); pitr != _priority.end(); ++pitr) {
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
        bool operator()(const std::pair<unsigned, mlir::Operation*>& left,
                        const std::pair<unsigned, mlir::Operation*>& right) const {
            unsigned priorityLeft = left.first;
            unsigned priorityRight = right.first;
            unsigned opIDLeft = getUniqueID(left.second).getInt();
            unsigned opIDright = getUniqueID(right.second).getInt();

            if (priorityLeft < priorityRight)
                return true;
            else if (priorityLeft > priorityRight)
                return false;
            else {
                return opIDLeft < opIDright;
            }
        }
    };

    // reassign the priority
    std::set<std::pair<unsigned, mlir::Operation*>, custom_compare> s;  // The new (temporary) container.
    for (auto const& pair : _priority)
        s.emplace(pair.second, pair.first);  // Flip the pairs.

    size_t newPriority = 1;
    for (auto const& pair : s) {
        _priority[pair.second] = newPriority++;
    }
}

void FeasibleBarrierScheduler::assignTaskUniqueIds() {
    int64_t uniqueId = 0;
    auto assignUniqueIDs = [&](VPURT::TaskOp taskOp) {
        taskOp->setAttr(uniqueIdAttrName, getIntAttr(_ctx, uniqueId++));
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

bool FeasibleBarrierScheduler::doesOpRunOnNCE(mlir::Operation* op) {
    if ((mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::NCE) ||
        (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::DMA_NN))
        return true;
    else
        return false;
}

unsigned FeasibleBarrierScheduler::countProducerConsumerTasks(mlir::Operation* op) {
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::NCE) {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
        auto& block = taskOp.body().getBlocks().front();
        auto wrappedTaskOp = block.begin();
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
        return 1;
    } else {
        VPUX_THROW("This operation does not run on hardware");
    }
}

void FeasibleBarrierScheduler::createTaskBarrierResourceUtilityTable() {
    for (auto& op : _inDegree) {
        if (doesOpRunOnNCE(op.first)) {
            auto resource_utility = countProducerConsumerTasks(op.first);
            _log.trace("Operation {0} is a DPU or DMA task and requires {1} barrier slots", getUniqueID(op.first),
                       resource_utility);
            _resourceUtilityMap.insert(std::make_pair(op.first, resource_utility));
        } else  // UPA tasks
        {
            _log.trace("Operation: {0} is a UPA tasks and requires 0 slots", getUniqueID(op.first));
            _resourceUtilityMap.insert(std::make_pair(op.first, 0));
        }
    }
}

void FeasibleBarrierScheduler::init() {
    _log.trace("Feasible barrier scheduler initialization");

    // Assing unique IDs to tasks
    assignTaskUniqueIds();

    // Save the original IR dependency, it may need to be restored
    saveOriginalIRDependency();

    // Retrieve output ops (ops with no out-degree)
    _outputOps = _outputOpsBackUp;

    // Assign task priorities
    computeTaskPriorities();

    // Store per-task barrier producer utilization
    createTaskBarrierResourceUtilityTable();
}

bool FeasibleBarrierScheduler::schedule(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    _processedTasks.clear();
    _scheduleableCandidates.clear();
    _scheduledTasks.clear();
    _barrierAssociationTable.clear();
    _barrierCount = numberOfBarriers;
    _slotsPerBarrier = maxProducersPerBarrier;
    _inDegree = _inDegreeBackUp;

    // retrieve output ops (ops with no out-degree)
    _outputOps = _outputOpsBackUp;

    // Create a barrier transition structure per barrier
    initializeBarrierAssociationTable();

    _log.trace("Initializing the barrier resource upper state");
    initializeBarrierResourceState(numberOfBarriers, maxProducersPerBarrier);

    operation_in_degree_t::iterator itr = _inDegree.begin();
    while (itr != _inDegree.end()) {
        auto op = itr->first;
        if (_inDegree.find(op)->second == 0) {
            _log.trace("Adding task: {0} to candidate set", getUniqueID(op));
            addTaskToCandidateSet(op);
        }
        ++itr;
    }

    VPUX_THROW_UNLESS(!_scheduleableCandidates.empty(),
                      "No operations with zero in-degree exist, error processing the dependencies");

    // Scheduling loop, loop until all output tasks are scheduled
    scheduleOperations();

    // Insert barriers in the IR based on the output of the list schedule
    insertBarriersinIR();

    //TODO John - this should not always be true
    return true;
}

void FeasibleBarrierScheduler::reorderIR() {
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

void FeasibleBarrierScheduler::insertBarriersinIR() {
    size_t scheduling_number = 0UL;
    size_t btask_count = 0UL;
    mlir::OpBuilder builder(_func.getBody());

    for (const auto& op : _scheduledTasks) {
        auto bitr = _barrierAssociationTable.find(op.barrier_index_);
        assert(bitr != _barrierAssociationTable.end());
        barrierTransitionStructure& bstructure = bitr->second;

        // Set scheduling number
        _log.trace("Assigning scheduling number {0} to the task {1} ", scheduling_number,
                   FeasibleBarrierScheduler::getUniqueID(op.op_));
        op.op_->setAttr(schedulingNumberAttrName, getIntAttr(_ctx, scheduling_number));

        scheduling_number++;

        // STEP-2: update barrier structure invariant //
        bool new_barrier_task_created = bstructure.processNextScheduledTask(op, builder);

        if (new_barrier_task_created) {
            ++btask_count;
        }
    }

    // STEP-2.5: process trailing barrier control structures //
    {
        for (auto bitr = _barrierAssociationTable.begin(); bitr != _barrierAssociationTable.end(); ++bitr) {
            barrierTransitionStructure& bstruct = bitr->second;
            bstruct.closeBarrierProducerList();
        }
    }

    // TODO Remove this logging when ready for review
    // for (auto& barrier : configureBarrierOpUpdateWaitMap) {
    //     _log.trace("Barrier ID {0} has the following producers", barrier.first->getAttr("id"));
    //     for (auto op : barrier.second.first)
    //         _log.trace("producer Op with ID {0} to barrier {1}", FeasibleBarrierScheduler::getUniqueID(op),
    //                    barrier.first->getAttr("id"));
    // }

    // for (auto& barrier : configureBarrierOpUpdateWaitMap) {
    //     _log.trace("Barrier ID {0} has the following consumers", barrier.first->getAttr("id"));
    //     for (auto op : barrier.second.second)
    //         _log.trace("consumer Op with ID {0} to barrier {1}", FeasibleBarrierScheduler::getUniqueID(op),
    //                    barrier.first->getAttr("id"));
    // }

    _log.trace("Barrier scheduling complete");

    getTaskOpUpdateWaitMap(configureBarrierOpUpdateWaitMap, configureTaskOpUpdateWaitMap);

    removeRedundantDependency();
    removeRedundantBarrier();

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
            assert(taskOp != NULL);
            assert(barrierOp.barrier() != NULL);
            _log.trace("Adding Barrier ID {0} as an update barrier for operation {1}", barrierOp->getAttr("id"),
                       FeasibleBarrierScheduler::getUniqueID(user));
            taskOp.updateBarriersMutable().append(barrierOp.barrier());
        }
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second.second) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
            assert(taskOp != NULL);
            assert(barrierOp.barrier() != NULL);
            _log.trace("Adding Barrier ID {0} as an wait barrier for operation {1}", barrierOp->getAttr("id"),
                       FeasibleBarrierScheduler::getUniqueID(user));
            taskOp.waitBarriersMutable().append(barrierOp.barrier());
        }
    }
}

void FeasibleBarrierScheduler::cleanUpVirtualBarriers() {
    _func->walk([](VPURT::TaskOp op) {
        op.updateBarriersMutable().clear();
        op.waitBarriersMutable().clear();
    });

    _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
        op->dropAllUses();
        op.erase();
    });
}

bool FeasibleBarrierScheduler::performRuntimeSimulation() {
    bool success = true;
    reorderIR();
    if (configureBarrierOpUpdateWaitMap.size()) {
        // run simulation
        VPURT::BarrierSimulator barrierSim(_func);
        VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");
        success = mlir::succeeded(barrierSim.simulateBarriers(_log.nest()));

        // if (_barrierCount == 4)
        //     success = false;
    }

    if (!success) {
        cleanUpVirtualBarriers();
        configureBarrierOpUpdateWaitMap.clear();
        configureTaskOpUpdateWaitMap.clear();
    }

    std::cout << "Barrier simualtion result is " << success << " with upperbound " << _barrierCount << std::endl;

    return success;
}

// If two barriers have same consumers, they can be merged
// If a barrier has no producers, it can be removed
void FeasibleBarrierScheduler::removeRedundantBarrier() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        auto consumers = (*iter).second.second;
        auto iter1 = iter;
        iter1++;
        for (; iter1 != configureBarrierOpUpdateWaitMap.end();) {
            auto consumers1 = (*iter1).second.second;
            if (consumers1 == consumers) {
                _log.trace("found barrier {0} and {1} have same consumers", (*iter).first->getAttr("id"),
                           (*iter1).first->getAttr("id"));
                auto producers = (*iter1).second.first;
                for (auto& task : producers) {
                    (*iter).second.first.insert(task);
                }
                auto removedIter = iter1;
                iter1++;
                (*removedIter).first->dropAllUses();
                (*removedIter).first->erase();
                configureBarrierOpUpdateWaitMap.erase(removedIter);
            } else
                iter1++;
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
void FeasibleBarrierScheduler::removeRedundantDependency() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        // producers
        auto producers = (*iter).second.first;
        for (auto prod = producers.begin(); prod != producers.end();) {
            auto prod1 = prod;
            prod1++;
            for (; prod1 != producers.end();) {
                if (isPathExist(*prod1, *prod)) {
                    auto removedIter = prod1;
                    prod1++;
                    producers.erase(removedIter);
                } else if (isPathExist(*prod, *prod1)) {
                    auto removedIter = prod;
                    prod++;
                    producers.erase(removedIter);
                    break;
                } else
                    prod1++;
            }
            if (prod1 == producers.end())
                prod++;
        }
        (*iter).second.first = producers;

        // consumers
        auto consumers = (*iter).second.second;
        for (auto cons = consumers.begin(); cons != consumers.end();) {
            auto cons1 = cons;
            cons1++;
            for (; cons1 != consumers.end();) {
                if (isPathExist(*cons, *cons1)) {
                    auto removedIter = cons1;
                    cons1++;
                    consumers.erase(removedIter);
                } else if (isPathExist(*cons1, *cons)) {
                    auto removedIter = cons;
                    cons++;
                    consumers.erase(removedIter);
                    break;
                } else
                    cons1++;
            }
            if (cons1 == consumers.end())
                cons++;
        }
        (*iter).second.second = consumers;
    }
}

void FeasibleBarrierScheduler::initializeBarrierAssociationTable() {
    _log.trace("STEP-0: Initialize the association table");
    for (size_t barrierId = 1; barrierId <= _barrierCount; barrierId++) {
        auto bitr =
                _barrierAssociationTable.insert(std::make_pair(barrierId, barrierTransitionStructure(*this)));
        barrierTransitionStructure& bstructure = (bitr.first)->second;
        bstructure.init();
    }
}

// detect if op b depends on a
bool FeasibleBarrierScheduler::isPathExist(mlir::Operation* a, mlir::Operation* b) {
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
                if (isPathExist(*consumer, b))
                    return true;
            }
        }
        return false;
    }
}

void FeasibleBarrierScheduler::populateScheduledOps(mlir::Operation* scheduledOp) {
    // populate the struct fields
    ScheduledOpInfo scheduled;

    scheduled.op_ = scheduledOp;
    scheduled.schedule_time_ = _currentTime;

    const resource_state_t& rstate = resourceState();
    const barrier_info_t& binfo = rstate.get_barrier_info(scheduledOp);

    scheduled.barrier_index_ = binfo.bindex_;
    scheduled.slot_count_ = binfo.slot_count_;

    // _log.trace("Get barrier info for operation {0}", FeasibleBarrierScheduler::getUniqueID(scheduledOp));

    // _log.trace("The time is {0}, the Operation is {1} end time is {1}", scheduled.schedule_time_,
    //                        FeasibleBarrierScheduler::getUniqueID(scheduledOp));
    // _log.trace("The barrier index is {0}, , the slot cout is {1}", scheduled.barrier_index_,
    //                        scheduled.slot_count_);
    // _log.trace("Pushing to _scheduledTasks, the time is {0}, the Operation is {1}",
    // scheduled.schedule_time_,
    //                       FeasibleBarrierScheduler::getUniqueID(scheduledOp));

    _scheduledTasks.push_back(scheduled);
}