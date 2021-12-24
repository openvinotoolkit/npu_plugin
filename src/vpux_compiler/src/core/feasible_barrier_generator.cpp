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

using namespace vpux;

static constexpr StringLiteral schedulingNumberAttrName = "SchedulingNumber";
std::map<mlir::Operation*, SmallVector<mlir::Operation*>> FeasibleBarrierScheduler::barrierProducersMap{};
std::map<mlir::Operation*, SmallVector<mlir::Operation*>> FeasibleBarrierScheduler::barrierConsumersMap{};

FeasibleBarrierScheduler::FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log)
        : _barrierCount(),
          _slotsPerBarrier(),
          _resource_state(),
          _in_degree(),
          _heap(),
          _current_time(0),
          _candidates(),
          _heap_ordering(),
          _schedulable_op(),
          _processed_ops(),
          _priority(),
          _log(log),
          _ctx(ctx),
          _func(func),
          builder(_func.getBody()) {
    saveOriginalDependency();
    _barrierOpUpdateWaitMap = configureBarrierOpUpdateWaitMapBackUp;
    _taskOpUpdateWaitMap = configureTaskOpUpdateWaitMapBackUp;
};

void FeasibleBarrierScheduler::getTaskOpUpdateWaitMap(
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                barrierOpUpdateWaitMap,
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                taskOpUpdateWaitMap) {
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

void FeasibleBarrierScheduler::getTaskOpUpdateWaitMap(
        std::map<mlir::Operation*,
                 std::pair<std::set<mlir::Operation*, task_operation_comparator_t>,
                           std::set<mlir::Operation*, task_operation_comparator_t>>,
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

void FeasibleBarrierScheduler::saveOriginalDependency() {
    configureBarrierOpUpdateWaitMapBackUp.clear();
    configureTaskOpUpdateWaitMapBackUp.clear();

    auto _barrierOps = to_small_vector(_func.getOps<VPURT::DeclareVirtualBarrierOp>());
    std::cout << "get initial barriers " << _barrierOps.size() << std::endl;
    for (auto& barrierOp : _barrierOps) {
        std::set<mlir::Operation*> producers{};
        std::set<mlir::Operation*> consumers{};

        for (auto* userOp : barrierOp->getUsers()) {
            std::cout << "get Users " << std::endl;
            auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);

            VPUX_THROW_WHEN(opEffects == nullptr,
                            "Barrier '{0}' is used by Operation '{1}' without MemoryEffects interface",
                            barrierOp->getLoc(), userOp->getName());

            using MemEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

            SmallVector<MemEffect> valEffects;

            opEffects.getEffectsOnValue(barrierOp.barrier(), valEffects);

            VPUX_THROW_WHEN(
                    valEffects.size() != 1,
                    "Barrier '{0}' must have exactly 1 MemoryEffect per Operation, got '{1}' for Operation '{2}'",
                    barrierOp->getLoc(), valEffects.size(), userOp->getLoc());

            const auto& effect = valEffects.front();

            std::cout << "detect producers and consumers " << std::endl;

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task == nullptr) {
                    exit(0);
                }
                // Logger::global().error("Task with scheduling number {0}", task->getAttr("SchedulingNumber"));
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    producers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    producers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    producers.insert(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task == nullptr) {
                    exit(0);
                }
                // Logger::global().error("Task with scheduling number {0}", task->getAttr("SchedulingNumber"));
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    consumers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    consumers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    consumers.insert(userOp);
                }
            } else {
                VPUX_THROW("Barrier '{0}' has unsupported Effect in Operation '{1}'", barrierOp->getLoc(),
                           userOp->getLoc());
            }

            std::cout << "finish" << std::endl;
        }
        configureBarrierOpUpdateWaitMapBackUp.insert(std::make_pair(barrierOp, std::make_pair(producers, consumers)));
    }

    getTaskOpUpdateWaitMap(configureBarrierOpUpdateWaitMapBackUp, configureTaskOpUpdateWaitMapBackUp);

    _func->walk([](VPURT::TaskOp op) {
        op.updateBarriersMutable().clear();
        op.waitBarriersMutable().clear();
    });

    _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
        op->dropAllUses();
        op.erase();
    });

    std::cout << "Removed all declare virtual barrier ops" << std::endl;
}

bool FeasibleBarrierScheduler::reached_end() const {
    return _candidates.empty() && _heap.empty();
}

bool FeasibleBarrierScheduler::operator==(const FeasibleBarrierScheduler& o) const {
    return reached_end() && o.reached_end();
}

void FeasibleBarrierScheduler::operator++() {
    // Logger::global().error("Calling operator++ for Feasible Schedule generator");
    // nextSchedulableOperation();
}

void FeasibleBarrierScheduler::pushToHeap(const HeapElement& elem) {
    _heap.push_back(elem);
    std::push_heap(_heap.begin(), _heap.end(), _heap_ordering);
}

FeasibleBarrierScheduler::HeapElement FeasibleBarrierScheduler::popFromHeap() {
    std::pop_heap(_heap.begin(), _heap.end(), _heap_ordering);
    HeapElement elem = _heap.back();
    _heap.pop_back();
    return elem;
}

void FeasibleBarrierScheduler::addToCandidateSet(mlir::Operation* op) {
    if (_processed_ops.find(op) != _processed_ops.end()) {
        return;
    }
    Logger::global().error("Adding operation  to candidates list {0} to candidates list", getUniqueID(op));
    _candidates.push_back(op);
    _processed_ops.insert(op);
}

void FeasibleBarrierScheduler::addOutGoingOperationsToCandidateList(mlir::Operation* op) {
    Logger::global().error("Add outgoing operations to candidate list");

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

        Logger::global().error("Decrementing the in-degree of operation {0}", getUniqueID(*itr));

        typename operation_in_degree_t::iterator deg_itr = _in_degree.find(op);

        VPUX_THROW_UNLESS((deg_itr != _in_degree.end()) && (deg_itr->second > 0), "Invalid indegree");
        // assert((deg_itr != _in_degree.end()) && (deg_itr->second > 0));

        if (deg_itr->second == 1) {
            Logger::global().error("Adding operation {0} to candidate_list", getUniqueID(*itr));
            addToCandidateSet(op);
            Logger::global().error("Erasing operation {0} from the in_degree table", getUniqueID(*itr));
            _in_degree.erase(deg_itr);
        } else {
            --(deg_itr->second);
        }
    }
}

bool FeasibleBarrierScheduler::scheduleOperations() {
    _schedulable_op = NULL;

    // scheduling loop, loop until all output ops are scheduled
    while (!_outputOps.empty()) {
        schedulable_ops_iterator_t op_itr = findSchedulableOp();

        if (isValidOp(op_itr)) {
            // found a schedulable operation //
            mlir::Operation* op = (*op_itr);

            delay_t op_delay = 1;
            resource_t op_resources = _resource_utility_map[*op_itr];
            schedule_time_t op_end_time = _current_time + op_delay;

            Logger::global().error("Operation {0} end time is {1} pushing to heap", getUniqueID(*op_itr), op_end_time);
            pushToHeap(HeapElement(op, op_end_time));

            _candidates.erase(op_itr);

            // schedule operation
            scheduleOperation(op, op_resources);

            _schedulable_op = op;
            populateScheduledOps(op);
            Logger::global().error("The _schedulable_op ID is {0}", getUniqueID(_schedulable_op));
            // decrease outputs ops if output op scheduled
            if (_outputOps.find(op) != _outputOps.end()) {
                _outputOps.erase(op);
            }

        } else if (!_heap.empty()) {
            // no-op found so move up the schedule time to the smallest completion
            // time among the active operations. //
            HeapElement top_elem = popFromHeap();
            mlir::Operation* op = top_elem.op_;

            // assert(_current_time <= top_elem.time_);
            VPUX_THROW_UNLESS(_current_time <= top_elem.time_, "Invalid indegree");
            _current_time = top_elem.time_;
            // since operation is now complete update the schedule //

            unScheduleOperation(op);
            // since op has completed add all out-going ops to candidates //
            addOutGoingOperationsToCandidateList(op);
        } else {
            // schedule is not feasible //
            _candidates.clear();
            break;
        }
    }

    // return _schedulable_op != NULL;
    return true;
}

bool FeasibleBarrierScheduler::isValidOp(schedulable_ops_iterator_t itr) const {
    return !(itr == _candidates.end());
}

FeasibleBarrierScheduler::schedulable_ops_iterator_t FeasibleBarrierScheduler::findSchedulableOp() {
    _log.trace("Looking for a a scheduleable operation");

    schedulable_ops_iterator_t itr = _candidates.end();
    std::list<schedulable_ops_iterator_t> ready_list;

    _log.trace("There are {0} candiates and for each candiate", _candidates.size());

    for (itr = _candidates.begin(); itr != _candidates.end(); ++itr) {
        _log.trace("The demand for operation {0} is {1}", getUniqueID(*itr), _resource_utility_map[*itr]);

        if (isResourceAvailable(_resource_utility_map[*itr])) {
            _log.trace("Adding operation {0} to the ready list", getUniqueID(*itr));
            ready_list.push_back(itr);
        }
    }

    _log.trace("Finding the operation with lowest priority in ready list");
    // find the one with lowest priority //
    if (!ready_list.empty()) {
        size_t min_priority = std::numeric_limits<size_t>::max();
        for (auto ritr = ready_list.begin(); ritr != ready_list.end(); ++ritr) {
            size_t curr_priority = _priority[*(*ritr)];
            if (curr_priority < min_priority) {
                itr = *ritr;
                min_priority = curr_priority;
            }
        }
    }
    return itr;
}

mlir::Operation*& FeasibleBarrierScheduler::operator*() {
    Logger::global().error("Calling FeasibleBarrierScheduler::operator*()");
    if (!_schedulable_op)
        std::runtime_error("Feasible_Schedule_Generator: Null ptr dereference");

    Logger::global().error("Returning operation {0}", getUniqueID(_schedulable_op));
    return _schedulable_op;
}

size_t FeasibleBarrierScheduler::currentTime() const {
    return _current_time;
}

const resource_state_t& FeasibleBarrierScheduler::resourceState() const {
    return _resource_state;
}

bool FeasibleBarrierScheduler::unScheduleOperation(mlir::Operation*& op) {
    return _resource_state.unschedule_operation(op);
}

bool FeasibleBarrierScheduler::scheduleOperation(mlir::Operation*& op, resource_t demand) {
    return _resource_state.schedule_operation(op, demand);
}

bool FeasibleBarrierScheduler::isResourceAvailable(const resource_t& demand) {
    return _resource_state.is_resource_available(demand);
}

void FeasibleBarrierScheduler::initResourceState(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    op_resource_state_t resource(numberOfBarriers, maxProducersPerBarrier);
    _resource_state.init(resource);
}

SmallVector<mlir::Operation*> FeasibleBarrierScheduler::getConsumerOps(mlir::Operation* op) {
    SmallVector<mlir::Operation*> consumerOps;
    if (auto task = mlir::dyn_cast<VPURT::TaskOp>(op)) {
        for (auto updateBarrier : _taskOpUpdateWaitMap[task].second) {
            // Logger::global().error("The operation has {0} consumers", barrierConsumersMap[updateBarrier].size());
            // Logger::global().error("The operation ID  {0} has {1} consumers ", getUniqueID(op),
            //                        barrierConsumersMap[updateBarrier].size());
            consumerOps.insert(consumerOps.end(), _barrierOpUpdateWaitMap.find(updateBarrier)->second.second.begin(),
                               _barrierOpUpdateWaitMap.find(updateBarrier)->second.second.end());
        }
    } else {
        exit(1);
    }
    return consumerOps;
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

void FeasibleBarrierScheduler::computeOperationPriorities() {
    operation_in_degree_t in_degree;

    computeOpIndegree(in_degree);

    // assign topological sort level as priority to start with //
    std::list<mlir::Operation*> zero_in_degree_nodes[2];
    _priority.clear();

    size_t curr_priority = 0;

    operation_in_degree_t::iterator itr = _in_degree.begin();
    operation_in_degree_t::iterator itr_end = _in_degree.end();

    while (itr != itr_end) {
        auto op = itr->first;
        if (_in_degree.find(op)->second == 0) {
            Logger::global().error("Adding op {0}  to zero_in_degree_nodes ", getUniqueID(op));
            zero_in_degree_nodes[curr_priority % 2].push_back(op);
            Logger::global().error("Priority for  op {0}  is {1}", getUniqueID(op), curr_priority);
            _priority[op] = curr_priority;
        }
        ++itr;
    }

    while (!zero_in_degree_nodes[curr_priority % 2].empty()) {
        // decrement the in-degree

        for (auto op = zero_in_degree_nodes[curr_priority % 2].begin();
             op != zero_in_degree_nodes[curr_priority % 2].end(); ++op) {
            auto opConsumers = getConsumerOps(*op);

            SmallVector<mlir::Operation*>::iterator jtr = opConsumers.begin();
            SmallVector<mlir::Operation*>::iterator jtr_end = opConsumers.end();

            while (jtr != jtr_end) {
                Logger::global().error("Looking up operation {0} in the in_degree table ", getUniqueID(*jtr));
                typename operation_in_degree_t::iterator deg_itr = in_degree.find(*jtr);

                VPUX_THROW_UNLESS((deg_itr != in_degree.end()) && (deg_itr->second > 0), "Invalid indegree");

                // assert((deg_itr != in_degree.end()) && (deg_itr->second > 0));
                std::cout << "Operation  has an indegree of " << deg_itr->second << std::endl;
                (deg_itr->second)--;
                std::cout << "Decrementing the in-degree of ther operation, the indegree is now " << deg_itr->second
                          << std::endl;

                if (!(deg_itr->second)) {
                    // in-degree of this node has become zero//
                    Logger::global().error("The in-degree of op operation {0}  has become zero ",
                                           getUniqueID(deg_itr->first));

                    Logger::global().error("The priority of op {0}  has become  {1} ", getUniqueID(deg_itr->first),
                                           (curr_priority + 1));

                    _priority[deg_itr->first] = (curr_priority + 1);
                    zero_in_degree_nodes[(curr_priority + 1) % 2].push_back(deg_itr->first);

                    Logger::global().error("Erasing op {0} from the in-degree table ", getUniqueID(deg_itr->first));
                    in_degree.erase(deg_itr);
                }
                ++jtr;
            }
        }
        zero_in_degree_nodes[curr_priority % 2].clear();
        ++curr_priority;
    }

    Logger::global().error("Printing priority map");
    for (auto const& pair : _priority) {
        Logger::global().error("{Operation {0}  priority {1}", getUniqueID(pair.first), pair.second);
    }

    for (typename priority_map_t::iterator pitr = _priority.begin(); pitr != _priority.end(); ++pitr) {
        Logger::global().error("Checking priority of {0} ", getUniqueID(pitr->first));
        auto opConsumers = getConsumerOps((pitr->first));
        // set priority to max of all out going priorities //
        SmallVector<mlir::Operation*>::iterator jtr = opConsumers.begin();
        SmallVector<mlir::Operation*>::iterator jtr_end = opConsumers.end();

        if (!(pitr->second)) {
            size_t max = pitr->second;
            while (jtr != jtr_end) {
                max = std::max(_priority[*jtr], max);
                ++jtr;
            }
            std::cout << "Setting the priority of " << /*printOpType(pitr->first) <<*/ " to " << max << std::endl;
            pitr->second = max;
        }
    }
    for (auto const& pair : _priority) {
        Logger::global().error("{Operation {0}  priority {1}", getUniqueID(pair.first), pair.second);
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

    for (auto const& pair : _priority) {
        Logger::global().error("{Operation {0}  priority {1}", getUniqueID(pair.first), pair.second);
    }
}

void FeasibleBarrierScheduler::assignUniqueIds() {
    int64_t uniqueId = 0;
    auto assignUniqueIDs = [&](VPURT::TaskOp taskOp) {
        taskOp->setAttr(uniqueIdAttrName, getIntAttr(_ctx, uniqueId++));
        std::cout << "Assigning ID " << uniqueId << " to operation " << printOpType(taskOp) << std::endl;
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

void FeasibleBarrierScheduler::getBarriersProducersAndConsumers() {
    // Get all producers and consumers of barriers (NCE,UPA, DMA) only

    for (auto& barrierOpConfig : _barrierOpUpdateWaitMap) {
        SmallVector<mlir::Operation*> producers(barrierOpConfig.second.first.begin(),
                                                barrierOpConfig.second.first.end());
        SmallVector<mlir::Operation*> consumers(barrierOpConfig.second.second.begin(),
                                                barrierOpConfig.second.second.end());
        barrierProducersMap.insert(std::make_pair(barrierOpConfig.first, producers));
        barrierConsumersMap.insert(std::make_pair(barrierOpConfig.first, consumers));
    }
}

void FeasibleBarrierScheduler::computeOpOutdegree(operation_out_degree_t& out_degree) {
    out_degree.clear();

    _func.walk([&](VPURT::TaskOp taskOp) {
        size_t updateBarrierIncomingEdges = 0;

        for (const auto updateBarrier : _taskOpUpdateWaitMap[taskOp].second) {
            updateBarrierIncomingEdges += _barrierOpUpdateWaitMap[updateBarrier].second.size();
        }
        Logger::global().error("The outdegree for the operation {0}  is {1}", getUniqueID(taskOp.getOperation()),
                               updateBarrierIncomingEdges);

        out_degree.insert(std::make_pair(taskOp.getOperation(), updateBarrierIncomingEdges));
    });
    std::cout << "The size of outdegree table is " << out_degree.size() << std::endl;
}

void FeasibleBarrierScheduler::computeOpIndegree(operation_in_degree_t& in_degree) {
    in_degree.clear();

    _func.walk([&](VPURT::TaskOp taskOp) {
        size_t waitBarrierIncomingEdges = 0;

        for (const auto waitBarrier : _taskOpUpdateWaitMap[taskOp].first) {
            waitBarrierIncomingEdges += _barrierOpUpdateWaitMap[waitBarrier].first.size();
        }
        Logger::global().error("The indegree for the operation {0}  is {1}", getUniqueID(taskOp.getOperation()),
                               waitBarrierIncomingEdges);

        in_degree.insert(std::make_pair(taskOp.getOperation(), waitBarrierIncomingEdges));
    });
    std::cout << "The size of indegree table is " << in_degree.size() << std::endl;
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

void FeasibleBarrierScheduler::createOperationResourceUtilityTable() {
    for (auto& op : _in_degree) {
        _log.trace("Operation: {0} ", getUniqueID(op.first));
        if (doesOpRunOnNCE(op.first)) {
            auto resource_utility = countProducerConsumerTasks(op.first);
            // resource utility //
            _log.trace("Operation: {0} uses {1} slots", getUniqueID(op.first), resource_utility);
            _resource_utility_map.insert(std::make_pair(op.first, resource_utility));
        } else  // UPA tasks
        {
            // resource utility is 0 //
            _log.trace("Operation: {0} uses 0 slots", getUniqueID(op.first));
            _resource_utility_map.insert(std::make_pair(op.first, 0));
        }
    }
}

void FeasibleBarrierScheduler::init() {
    _log.trace("Feasible Barrier Scheduler initialization");

    assignUniqueIds();
    getBarriersProducersAndConsumers();
    computeOpIndegree(_in_degree);
    computeOpOutdegree(_out_degree);

    // retrieve output ops (ops with no out-degree)
    for (auto& entry : _out_degree) {
        if (entry.second == 0) {
            _outputOps.insert(entry.first);
        }
    }

    createOperationResourceUtilityTable();
}

bool FeasibleBarrierScheduler::schedule(size_t numberOfBarriers, size_t maxProducersPerBarrier) {
    _priority.clear();
    _processed_ops.clear();
    _candidates.clear();
    _scheduledOps.clear();
    _barrierAssociationTable.clear();
    _barrierCount = numberOfBarriers;
    _slotsPerBarrier = maxProducersPerBarrier;

    computeOpIndegree(_in_degree);

    // retrieve output ops (ops with no out-degree)
    for (auto& entry : _out_degree) {
        if (entry.second == 0) {
            _outputOps.insert(entry.first);
        }
    }

    initializeBarrierAssociationTable();

    _log.trace("Initializing the resource upper state");
    initResourceState(numberOfBarriers, maxProducersPerBarrier);

    operation_in_degree_t::iterator itr = _in_degree.begin();
    operation_in_degree_t::iterator itr_end = _in_degree.end();

    while (itr != itr_end) {
        auto op = itr->first;
        if (_in_degree.find(op)->second == 0) {
            _log.trace("Adding op: {0} to candidate set", getUniqueID(op));
            addToCandidateSet(op);
        }
        ++itr;
    }

    VPUX_THROW_UNLESS(!_candidates.empty(),
                      "No operations with zero in-degree exist, error processing the dependencies");

    computeOperationPriorities();

    scheduleOperations();
    insertBarriersinIR();
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

    for (const auto& op : _scheduledOps) {
        auto bitr = _barrierAssociationTable.find(op.barrier_index_);
        assert(bitr != _barrierAssociationTable.end());
        barrierTransitionStructure& bstructure = bitr->second;

        // Set scheduling number
        Logger::global().error("Assigning scheduling number {0} to the Operation {1} ", scheduling_number,
                               FeasibleBarrierScheduler::getUniqueID(op.op_));
        op.op_->setAttr(schedulingNumberAttrName, getIntAttr(_ctx, scheduling_number));

        scheduling_number++;

        // STEP-2: update barrier structure invariant //
        bool new_barrier_task_created = bstructure.process_next_scheduled_op(op, builder);

        if (new_barrier_task_created) {
            ++btask_count;
        }
    }

    // STEP-2.5: process trailing barrier control structures //
    {
        for (auto bitr = _barrierAssociationTable.begin(); bitr != _barrierAssociationTable.end(); ++bitr) {
            barrierTransitionStructure& bstruct = bitr->second;
            bstruct.close_barrier_producer_list();
        }
    }

    // update,wait
    for (auto& barrier : configureBarrierOpUpdateWaitMap) {
        Logger::global().error("Barrier ID {0} has the following producers", barrier.first->getAttr("id"));
        for (auto op : barrier.second.first)
            Logger::global().error("producer Op with ID {0} to barrier {1}", FeasibleBarrierScheduler::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    for (auto& barrier : configureBarrierOpUpdateWaitMap) {
        Logger::global().error("Barrier ID {0} has the following consumers", barrier.first->getAttr("id"));
        for (auto op : barrier.second.second)
            Logger::global().error("consumer Op with ID {0} to barrier {1}", FeasibleBarrierScheduler::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    std::cout << "Done scheduling" << std::endl;

    getTaskOpUpdateWaitMap(configureBarrierOpUpdateWaitMap, configureTaskOpUpdateWaitMap);

    std::cout << "Done creating configureTaskOpUpdateWaitMap" << std::endl;

    removeRedundantDependency();
    removeRedundantBarrier();

    for (auto itr = configureBarrierOpUpdateWaitMap.begin(); itr != configureBarrierOpUpdateWaitMap.end();) {
        if (itr->second.second.empty()) {
            Logger::global().error("Earsing virtual Barrier ID {0} as it has no producers", itr->first->getAttr("id"));
            (*itr).first->dropAllUses();
            (*itr).first->erase();
            itr = configureBarrierOpUpdateWaitMap.erase(itr);
        } else {
            ++itr;
        }
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        Logger::global().error("Virtual Barrier ID {0} has {1} consumers", barrierOp->getAttr("id"),
                               p.second.second.size());
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        Logger::global().error("Virtual Barrier ID {0} has {1} producers", barrierOp->getAttr("id"),
                               p.second.first.size());
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second.first) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
            assert(taskOp != NULL);
            assert(barrierOp.barrier() != NULL);
            Logger::global().error("Adding Barrier ID {0} as an update barrier for operation {1}",
                                   barrierOp->getAttr("id"), FeasibleBarrierScheduler::getUniqueID(user));
            taskOp.updateBarriersMutable().append(barrierOp.barrier());
        }
    }

    for (const auto& p : configureBarrierOpUpdateWaitMap) {
        auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
        for (auto* user : p.second.second) {
            auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
            assert(taskOp != NULL);
            assert(barrierOp.barrier() != NULL);
            Logger::global().error("Adding Barrier ID {0} as an wait barrier for operation {1}",
                                   barrierOp->getAttr("id"), FeasibleBarrierScheduler::getUniqueID(user));
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
    if (configureBarrierOpUpdateWaitMap.size()) {
        // run simulation
        VPURT::BarrierSimulator barrierSim(_func);
        VPUX_THROW_UNLESS(barrierSim.isDynamicBarriers(), "Barrier generated by barrier scheduler must be dynamic");
        success = mlir::succeeded(barrierSim.simulateBarriers(_log.nest()));

        // if (barrierCount_ == 4)
        //     success = false;
    }

    if (!success) {
        cleanUpVirtualBarriers();
        configureBarrierOpUpdateWaitMap.clear();
        configureTaskOpUpdateWaitMap.clear();
    }

    std::cout << "Barrier simualtion result is " << success << " with upperbound " << _barrierCount << std::endl;

    // reorderIR();
    return success;
}

void FeasibleBarrierScheduler::removeRedundantBarrier() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        auto consumers = (*iter).second.second;
        auto iter1 = iter;
        iter1++;
        for (; iter1 != configureBarrierOpUpdateWaitMap.end();) {
            auto consumers1 = (*iter1).second.second;
            if (consumers1 == consumers) {
                Logger::global().error("found barrier {0} and {1} have same consumers", (*iter).first->getAttr("id"),
                                       (*iter1).first->getAttr("id"));
                auto producers = (*iter1).second.first;
                // auto mergedProducers = (*iter).second.first
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
}

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
                    // configureTaskOpUpdateWaitMap[*prod1].second.erase((*iter).first);
                } else if (isPathExist(*prod, *prod1)) {
                    auto removedIter = prod;
                    prod++;
                    producers.erase(removedIter);
                    // configureTaskOpUpdateWaitMap[*prod].second.erase((*iter).first);
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
                    // configureTaskOpUpdateWaitMap[*cons1].first.erase((*iter).first);
                } else if (isPathExist(*cons1, *cons)) {
                    auto removedIter = cons;
                    cons++;
                    consumers.erase(removedIter);
                    // configureTaskOpUpdateWaitMap[*cons].first.erase((*iter).first);
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
    Logger::global().error("STEP-0: Initialize the association table");
    for (size_t barrierId = 1; barrierId <= _barrierCount; barrierId++) {
        auto bitr =
                _barrierAssociationTable.insert(std::make_pair(barrierId, barrierTransitionStructure(_func, *this)));
        barrierTransitionStructure& bstructure = (bitr.first)->second;
        bstructure.init();
    }
}

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
    scheduled.schedule_time_ = _current_time;

    const resource_state_t& rstate = resourceState();
    const barrier_info_t& binfo = rstate.get_barrier_info(scheduledOp);

    scheduled.barrier_index_ = binfo.bindex_;
    scheduled.slot_count_ = binfo.slot_count_;

    // Logger::global().error("Get barrier info for operation {0}", FeasibleBarrierScheduler::getUniqueID(scheduledOp));

    // Logger::global().error("The time is {0}, the Operation is {1} end time is {1}", scheduled.schedule_time_,
    //                        FeasibleBarrierScheduler::getUniqueID(scheduledOp));
    // Logger::global().error("The barrier index is {0}, , the slot cout is {1}", scheduled.barrier_index_,
    //                        scheduled.slot_count_);
    // Logger::global().error("Pushing to _scheduledOps, the time is {0}, the Operation is {1}",
    // scheduled.schedule_time_,
    //                       FeasibleBarrierScheduler::getUniqueID(scheduledOp));

    _scheduledOps.push_back(scheduled);
}