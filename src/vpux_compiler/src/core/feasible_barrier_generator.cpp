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
//#include "vpux/compiler/core/barrier_schedule_generator.hpp"

using namespace vpux;

std::map<mlir::Operation*, SmallVector<mlir::Operation*>> FeasibleBarrierScheduler::barrierProducersMap{};
std::map<mlir::Operation*, SmallVector<mlir::Operation*>> FeasibleBarrierScheduler::barrierConsumersMap{};

static constexpr StringLiteral uniqueIdAttrName = "uniqueId";
// FeasibleBarrierScheduler::FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func/*,
//                                                    const resource_state_t& rstate, Logger log*/)
//         : _log(log),
//           _ctx(ctx),
//           _func(func),
//           _in_degree(),
//           _heap(),
//           _current_time(0),
//           _candidates(),
//           _resource_state(),
//           _heap_ordering(),
//           _schedulable_op(),
//           _processed_ops(),
//           _priority(){
//                   // init(rstate);
//           };

FeasibleBarrierScheduler::FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log,
                                                   size_t numberOfBarriers, size_t maxProducersPerBarrier)
        : _barrierCount(numberOfBarriers),
          _slotsPerBarrier(maxProducersPerBarrier),
          _resource_state(numberOfBarriers, maxProducersPerBarrier),
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
          _func(func) {
    init();
};

// FeasibleBarrierScheduler::FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log)
//         : _log(log),
//           _ctx(ctx),
//           _func(func),
//           _in_degree(),
//           _heap(),
//           _current_time(0),
//           _candidates(),
//           _resource_state(),
//           _heap_ordering(),
//           _schedulable_op(),
//           _processed_ops(),
//           _priority(){};

bool FeasibleBarrierScheduler::reached_end() const {
    return _candidates.empty() && _heap.empty();
}

bool FeasibleBarrierScheduler::operator==(const FeasibleBarrierScheduler& o) const {
    return reached_end() && o.reached_end();
}

void FeasibleBarrierScheduler::operator++() {
    Logger::global().error("Calling operator++ for Feasible Schedule generator");
    nextSchedulableOperation();
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

bool FeasibleBarrierScheduler::nextSchedulableOperation() {
    _schedulable_op = NULL;

    do {
        schedulable_ops_iterator_t op_itr = find_schedulable_op();

        if (isValidOp(op_itr)) {
            // found a schedulable operation //
            mlir::Operation* op = (*op_itr);

            delay_t op_delay = 1;
            // resource_t op_resources = _resource_utility_map[*op_itr];
            schedule_time_t op_end_time = _current_time + op_delay;

            Logger::global().error("Operation {0} end time is {1} pushing to heap", getUniqueID(*op_itr), op_end_time);
            pushToHeap(HeapElement(op, op_end_time));

            _candidates.erase(op_itr);

            // vpux::BarrierScheduleGenerator::barrier_scheduler_traits::schedule_operation(op, op_resources,
            // _resource_state);

            _schedulable_op = op;
            populateScheduledOps(op);
            Logger::global().error("The _schedulable_op ID is {0}", getUniqueID(_schedulable_op));

        } else if (!_heap.empty()) {
            // no-op found so move up the schedule time to the smallest completion
            // time among the active operations. //
            HeapElement top_elem = popFromHeap();
            mlir::Operation* op = top_elem.op_;

            // assert(_current_time <= top_elem.time_);
            VPUX_THROW_UNLESS(_current_time <= top_elem.time_, "Invalid indegree");
            _current_time = top_elem.time_;
            // since operation is now complete update the schedule //

            // BarrierScheduleGenerator::barrier_scheduler_traits::unschedule_operation(op, _resource_state);
            // since op has completed add all out-going ops to candidates //
            addOutGoingOperationsToCandidateList(op);
        } else {
            // schedule is not feasible //
            _candidates.clear();
            break;
        }
    } while (!_schedulable_op && !reached_end());

    return _schedulable_op != NULL;
}

bool FeasibleBarrierScheduler::isValidOp(schedulable_ops_iterator_t itr) const {
    return !(itr == _candidates.end());
}

FeasibleBarrierScheduler::schedulable_ops_iterator_t FeasibleBarrierScheduler::find_schedulable_op() {
    Logger::global().error("Looking for a a scheduleable operation");

    schedulable_ops_iterator_t itr = _candidates.end();
    std::list<schedulable_ops_iterator_t> ready_list;

    Logger::global().error("There are {0} candiates and for each candiate", _candidates.size());

    for (itr = _candidates.begin(); itr != _candidates.end(); ++itr) {
        Logger::global().error("The demand for operation {0} is {1}", getUniqueID(*itr), _resource_utility_map[*itr]);

        // if
        // (vpux::BarrierScheduleGenerator::barrier_scheduler_traits::is_resource_available(_resource_utility_map[*itr],
        // _resource_state)) {

        //     Logger::global().error("Adding operation {0} to the ready list", getUniqueID(*itr));
        //     ready_list.push_back(itr);
        // }
    }

    Logger::global().error("Finding the operation with lowest priority in ready list");
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

// void FeasibleBarrierScheduler::initResourceState(const resource_state_t& start_state) {
//     vpux::BarrierScheduleGenerator::barrier_scheduler_traits::initialize_resource_state(start_state,
//     _resource_state);
// }

void FeasibleBarrierScheduler::initResourceState() {
    _resource_state.init(_resource_state);
}

SmallVector<mlir::Operation*> FeasibleBarrierScheduler::getConsumerOps(mlir::Operation* op) {
    SmallVector<mlir::Operation*> consumerOps;
    if (auto task = mlir::dyn_cast<VPURT::TaskOp>(op)) {
        for (auto updateBarrier : task.updateBarriers()) {
            if (auto barrierOp = updateBarrier.getDefiningOp()) {
                Logger::global().error("The operation has {0} consumers", barrierConsumersMap[barrierOp].size());
                Logger::global().error("The operation ID  {0} has {1} consumers ", getUniqueID(op),
                                       barrierConsumersMap[barrierOp].size());
                consumerOps.insert(consumerOps.end(), barrierConsumersMap.find(barrierOp)->second.begin(),
                                   barrierConsumersMap.find(barrierOp)->second.end());
            }
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
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getExecutorKind());
        }
    });
}

void FeasibleBarrierScheduler::getBarriersProducersAndConsumers() {
    _allBarrierOps = to_small_vector(_func.getOps<VPURT::DeclareVirtualBarrierOp>());

    for (auto& barrierOp : _allBarrierOps) {
        SmallVector<mlir::Operation*> producers;
        SmallVector<mlir::Operation*> consumers;

        for (auto* userOp : barrierOp->getUsers()) {
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

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    producers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    producers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    producers.push_back(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    consumers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    consumers.push_back(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    consumers.push_back(userOp);
                }
            } else {
                VPUX_THROW("Barrier '{0}' has unsupported Effect in Operation '{1}'", barrierOp->getLoc(),
                           userOp->getLoc());
            }
        }
        barrierProducersMap.insert(std::make_pair(barrierOp, producers));
        barrierConsumersMap.insert(std::make_pair(barrierOp, consumers));
    }
}

void FeasibleBarrierScheduler::computeOpIndegree(operation_in_degree_t& in_degree) {
    in_degree.clear();

    _func.walk([&](VPURT::TaskOp taskOp) {
        size_t waitBarrierIncomingEdges = 0;

        for (const auto waitBarrier : taskOp.waitBarriers()) {
            if (auto barrierOp = waitBarrier.getDefiningOp()) {
                waitBarrierIncomingEdges += barrierProducersMap[barrierOp].size();
            }
        }
        Logger::global().error("The indegree for the operation {0}  is {1}", getUniqueID(taskOp.getOperation()),
                               waitBarrierIncomingEdges);

        in_degree.insert(std::make_pair(taskOp.getOperation(), waitBarrierIncomingEdges));
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

void FeasibleBarrierScheduler::createOperationResourceUtilityTable() {
    for (auto& op : _in_degree) {
        Logger::global().error("Operation: {0} ", getUniqueID(op.first));
        if (doesOpRunOnNCE(op.first)) {
            auto resource_utility = countProducerConsumerTasks(op.first);
            // resource utility //
            Logger::global().error("Operation: {0} uses {1} slots", getUniqueID(op.first), resource_utility);
            _resource_utility_map.insert(std::make_pair(op.first, resource_utility));
        } else  // UPA tasks
        {
            // resource utility is 0 //
            Logger::global().error("Operation: {0} uses 0 slots", getUniqueID(op.first));
            _resource_utility_map.insert(std::make_pair(op.first, 0));
        }
    }
}

bool FeasibleBarrierScheduler::init() {
    _log.trace("Feasible Barrier Scheduler initialization");

    assignUniqueIds();
    getBarriersProducersAndConsumers();
    computeOpIndegree(_in_degree);
    createOperationResourceUtilityTable();
    initializeBarrierAssociationTable();

    _log.trace("Initializing the resource upper state");
    initResourceState();  // Clean this up

    // collect the ones with zero-in degree into candidates //
    _candidates.clear();

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

    return nextSchedulableOperation();
}

size_t FeasibleBarrierScheduler::schedule() {
    return true;
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

void FeasibleBarrierScheduler::removeRedundantWaitBarriers() {
}

void FeasibleBarrierScheduler::removeRedundantDependencies() {
}

void FeasibleBarrierScheduler::populateScheduledOps(mlir::Operation* scheduledOp) {
    // populate the struct fields
    ScheduledOpInfo scheduled;

    scheduled.op_ = scheduledOp;
    scheduled.schedule_time_ = _current_time;
    // const barrier_info_t& binfo = rstate.get_barrier_info(sinfo_.op_);

    // sinfo_.op_ = *scheduler_begin_;
    // sinfo_.schedule_time_ = scheduler_begin_.currentTime();
    // const resource_state_t& rstate = scheduler_begin_.resourceState();
    // Logger::global().error("Get barrier info for operation {0}", FeasibleBarrierScheduler::getUniqueID(sinfo_.op_));
    // const barrier_info_t& binfo = rstate.get_barrier_info(sinfo_.op_);
    // sinfo_.barrier_index_ = binfo.bindex_;
    // sinfo_.slot_count_ = binfo.slot_count_;

    // scheduled.op_ = scheduledOp.op_;
    // scheduled.opType_ = scheduledOp.opType_;
    // scheduled.time_ = scheduledOp.time_;
    // scheduled.resourceInfo_ = intervals;
    Logger::global().error("Pushing to _scheduledOps, the time is {0}, the Operation is {1}", scheduled.schedule_time_,
                           FeasibleBarrierScheduler::getUniqueID(scheduledOp));
    _scheduledOps.push_back(scheduled);
}