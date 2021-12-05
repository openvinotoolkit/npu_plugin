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

#include "vpux/compiler/core/feasible_schedule_generator.hpp"
#include "vpux/compiler/core/barrier_schedule_generator.hpp"

using namespace vpux;

std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>> FeasibleScheduleGenerator::barrierProducersMap{};
std::unordered_map<mlir::Operation*, SmallVector<mlir::Operation*>> FeasibleScheduleGenerator::barrierConsumersMap{};

static constexpr StringLiteral uniqueIdAttrName = "uniqueId";
FeasibleScheduleGenerator::FeasibleScheduleGenerator(mlir::MLIRContext* ctx, mlir::FuncOp func,
                                                     const resource_state_t& rstate)
        : _ctx(ctx),
          _func(func),
          heap_(),
          current_time_(0),
          candidates_(),
          resource_state_(),
          heap_ordering_(),
          schedulable_op_(),
          in_degree_(),
          processed_ops_(),
          priority_() {
    init(rstate);
};

FeasibleScheduleGenerator::FeasibleScheduleGenerator(mlir::MLIRContext* ctx, mlir::FuncOp func)
        : _ctx(ctx),
          _func(func),
          heap_(),
          current_time_(0),
          candidates_(),
          resource_state_(),
          heap_ordering_(),
          schedulable_op_(),
          in_degree_(),
          processed_ops_(),
          priority_(){};

bool FeasibleScheduleGenerator::reached_end() const {
    Logger::global().error("Schedule reached end True/False {0}", (candidates_.empty() && heap_.empty()));
    return candidates_.empty() && heap_.empty();
}

bool FeasibleScheduleGenerator::operator==(const FeasibleScheduleGenerator& o) const {
    return reached_end() && o.reached_end();
}

void FeasibleScheduleGenerator::operator++() {
    Logger::global().error("Calling operator++ for Feasible Schedule generator");
    next_schedulable_operation();
}

void FeasibleScheduleGenerator::pushToHeap(const heap_element_t& elem) {
    heap_.push_back(elem);
    std::push_heap(heap_.begin(), heap_.end(), heap_ordering_);
}

FeasibleScheduleGenerator::heap_element_t FeasibleScheduleGenerator::popFromHeap() {
    std::pop_heap(heap_.begin(), heap_.end(), heap_ordering_);
    heap_element_t elem = heap_.back();
    heap_.pop_back();
    return elem;
}

void FeasibleScheduleGenerator::add_to_candidate_set(mlir::Operation* op) {
    if (processed_ops_.find(op) != processed_ops_.end()) {
        return;
    }
    Logger::global().error("Adding operation  to candidates list {0} to candidates list", getUniqueID(op));
    candidates_.push_back(op);
    processed_ops_.insert(op);
}

void FeasibleScheduleGenerator::add_outgoing_operations_to_candidate_list(mlir::Operation* op) {
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

        typename operation_in_degree_t::iterator deg_itr = in_degree_.find(op);
        assert((deg_itr != in_degree_.end()) && (deg_itr->second > 0));

        if (deg_itr->second == 1) {
            Logger::global().error("Adding operation {0} to candidate_list", getUniqueID(*itr));
            add_to_candidate_set(op);
            Logger::global().error("Erasing operation {0} from the in_degree table", getUniqueID(*itr));
            in_degree_.erase(deg_itr);
        } else {
            --(deg_itr->second);
        }
    }
}

bool FeasibleScheduleGenerator::next_schedulable_operation() {
    schedulable_op_ = NULL;

    do {
        schedulable_ops_iterator_t op_itr = find_schedulable_op();

        if (is_valid_op(op_itr)) {
            // found a schedulable operation //
            mlir::Operation* op = (*op_itr);

            delay_t op_delay = 1;
            resource_t op_resources = resource_utility_map_[*op_itr];
            schedule_time_t op_end_time = current_time_ + op_delay;

            Logger::global().error("Operation {0} end time is {1} pushing to heap", getUniqueID(*op_itr), op_end_time);
            pushToHeap(heap_element_t(op, op_end_time));
            candidates_.erase(op_itr);

            vpux::BarrierScheduleGenerator::barrier_scheduler_traits::schedule_operation(op, op_resources,
                                                                                         resource_state_);
            schedulable_op_ = op;
            Logger::global().error("The schedulable_op_ ID is {0}", getUniqueID(schedulable_op_));

        } else if (!heap_.empty()) {
            // no-op found so move up the schedule time to the smallest completion
            // time among the active operations. //
            heap_element_t top_elem = popFromHeap();
            mlir::Operation* op = top_elem.op_;

            assert(current_time_ <= top_elem.time_);
            current_time_ = top_elem.time_;
            // since operation is now complete update the schedule //

            BarrierScheduleGenerator::barrier_scheduler_traits::unschedule_operation(op, resource_state_);
            // since op has completed add all out-going ops to candidates //
            add_outgoing_operations_to_candidate_list(op);
        } else {
            // schedule is not feasible //
            candidates_.clear();
            break;
        }
    } while (!schedulable_op_ && !reached_end());

    // Logger::global().error("Returning Op is schedulable, the schedulable_op_ ID is {0}",
    // getUniqueID(schedulable_op_));
    return schedulable_op_ != NULL;
}

bool FeasibleScheduleGenerator::is_valid_op(schedulable_ops_iterator_t itr) const {
    return !(itr == candidates_.end());
}

FeasibleScheduleGenerator::schedulable_ops_iterator_t FeasibleScheduleGenerator::find_schedulable_op() {
    Logger::global().error("Looking for a a scheduleable operation");

    schedulable_ops_iterator_t itr = candidates_.end();
    std::list<schedulable_ops_iterator_t> ready_list;

    Logger::global().error("There are {0} candiates and for each candiate", candidates_.size());

    for (itr = candidates_.begin(); itr != candidates_.end(); ++itr) {
        Logger::global().error("The demand for operation {0} is {1}", getUniqueID(*itr), resource_utility_map_[*itr]);
        if (vpux::BarrierScheduleGenerator::barrier_scheduler_traits::is_resource_available(resource_utility_map_[*itr],
                                                                                            resource_state_)) {
            Logger::global().error("Adding operation {0} to the ready list", getUniqueID(*itr));
            ready_list.push_back(itr);
        }
    }

    Logger::global().error("Finding the operation with lowest priority in ready list");
    // find the one with lowest priority //
    if (!ready_list.empty()) {
        size_t min_priority = std::numeric_limits<size_t>::max();
        for (auto ritr = ready_list.begin(); ritr != ready_list.end(); ++ritr) {
            size_t curr_priority = priority_[*(*ritr)];
            if (curr_priority < min_priority) {
                itr = *ritr;
                min_priority = curr_priority;
            }
        }
    }
    // Logger::global().error("Returning operation ID {0} as a schedulable op", getUniqueID(*itr));
    return itr;
}

mlir::Operation*& FeasibleScheduleGenerator::operator*() {
    Logger::global().error("Calling FeasibleScheduleGenerator::operator*()");
    if (!schedulable_op_)
        std::runtime_error("Feasible_Schedule_Generator: Null ptr dereference");

    Logger::global().error("Returning operation {0}", getUniqueID(schedulable_op_));
    return schedulable_op_;
}

size_t FeasibleScheduleGenerator::current_time() const {
    return current_time_;
}

const resource_state_t& FeasibleScheduleGenerator::resource_state() const {
    return resource_state_;
}

void FeasibleScheduleGenerator::init_resource_state(const resource_state_t& start_state) {
    vpux::BarrierScheduleGenerator::barrier_scheduler_traits::initialize_resource_state(start_state, resource_state_);
}

SmallVector<mlir::Operation*> FeasibleScheduleGenerator::getConsumerOps(mlir::Operation* op) {
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

std::string FeasibleScheduleGenerator::printOpType(VPURT::TaskOp task) {
    if (task.getTaskType() == VPUIP::TaskType::NCE2)
        return "NCE task";
    if (task.getTaskType() == VPUIP::TaskType::NNDMA)
        return "DMA task ";
    if (task.getTaskType() == VPUIP::TaskType::UPA)
        return "Upa task ";
}

mlir::IntegerAttr FeasibleScheduleGenerator::getUniqueID(mlir::Operation* op) {
    if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op))
        return taskOp->getAttr(uniqueIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
}

void FeasibleScheduleGenerator::compute_operation_priorities() {
    operation_in_degree_t in_degree;

    compute_op_indegree(in_degree);

    // assign topological sort level as priority to start with //
    std::list<mlir::Operation*> zero_in_degree_nodes[2];
    priority_.clear();

    size_t curr_priority = 0;

    operation_in_degree_t::iterator itr = in_degree_.begin();
    operation_in_degree_t::iterator itr_end = in_degree_.end();

    while (itr != itr_end) {
        auto op = itr->first;
        if (in_degree_.find(op)->second == 0) {
            Logger::global().error("Adding op {0}  to zero_in_degree_nodes ", getUniqueID(op));
            zero_in_degree_nodes[curr_priority % 2].push_back(op);
            Logger::global().error("Priority for  op {0}  is {1}", getUniqueID(op), curr_priority);
            priority_[op] = curr_priority;
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

                    priority_[deg_itr->first] = (curr_priority + 1);
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
    for (auto const& pair : priority_) {
        Logger::global().error("{Operation {0}  priority {1}", getUniqueID(pair.first), pair.second);
    }

    for (typename priority_map_t::iterator pitr = priority_.begin(); pitr != priority_.end(); ++pitr) {
        Logger::global().error("Checking priority of {0} ", getUniqueID(pitr->first));
        auto opConsumers = getConsumerOps((pitr->first));
        // set priority to max of all out going priorities //
        SmallVector<mlir::Operation*>::iterator jtr = opConsumers.begin();
        SmallVector<mlir::Operation*>::iterator jtr_end = opConsumers.end();

        if (!(pitr->second)) {
            size_t max = pitr->second;
            while (jtr != jtr_end) {
                max = std::max(priority_[*jtr], max);
                ++jtr;
            }
            std::cout << "Setting the priority of " << /*printOpType(pitr->first) <<*/ " to " << max << std::endl;
            pitr->second = max;
        }
    }
    std::cout << "Printing priority map " << std::endl;
    for (auto const& pair : priority_) {
        Logger::global().error("{Operation {0}  priority {1}", getUniqueID(pair.first), pair.second);
    }
    std::cout << "Finished compute_operation_priorities " << std::endl;
}

void FeasibleScheduleGenerator::assignUniqueIds(mlir::FuncOp func) {
    int64_t uniqueId = 0;
    auto assignUniqueIDs = [&](VPURT::TaskOp taskOp) {
        taskOp->setAttr(uniqueIdAttrName, getIntAttr(_ctx, uniqueId++));
        std::cout << "Assigning ID " << uniqueId << " to operation "
                  << printOpType(taskOp) << std::endl;
    };

    _func.walk([&](VPURT::TaskOp taskOp) {
        switch (taskOp.getTaskType()) {
        case VPUIP::TaskType::UPADMA: {
            assignUniqueIDs(taskOp);
        }
        case VPUIP::TaskType::NNDMA: {
            assignUniqueIDs(taskOp);
            break;
        }
        case VPUIP::TaskType::NCE2: {
            assignUniqueIDs(taskOp);
            break;
        }
        case VPUIP::TaskType::UPA: {
            assignUniqueIDs(taskOp);
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getTaskType());
        }
    });
}

void FeasibleScheduleGenerator::printInfo(mlir::FuncOp func) {
    auto getTaskInfo = [&](VPURT::TaskOp taskOp, const int64_t count = 1) {
        std::cout << printOpType(taskOp) << " # wait barriers " << taskOp.waitBarriers().size()
                  << std::endl;
    };

    func.walk([&](VPURT::TaskOp taskOp) {
        switch (taskOp.getTaskType()) {
        case VPUIP::TaskType::UPADMA:
        case VPUIP::TaskType::NNDMA: {
            getTaskInfo(taskOp);
            break;
        }
        case VPUIP::TaskType::NCE2: {
            getTaskInfo(taskOp);
            break;
        }
        case VPUIP::TaskType::UPA: {
            getTaskInfo(taskOp);
            break;
        }
        default:
            VPUX_THROW("Unsupported task type '{0}'", taskOp.getTaskType());
        }
    });
}

void FeasibleScheduleGenerator::getAllBarriersProducersAndConsumers() {
    // Get all producers and consumers of barriers (NCE,UPA, DMA) only
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

            // VPUX_THROW_WHEN(effect.getResource() != VPUIP::BarrierResource::get(),
            //                 "Barrier '{0}' has non Barrier Resource for Operation '{1}'", barrierOp->getLoc(),
            //                 userOp->getLoc());

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getTaskType() == VPUIP::TaskType::NCE2) {
                    producers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::NNDMA) {
                    producers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::UPA) {
                    producers.push_back(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task.getTaskType() == VPUIP::TaskType::NCE2) {
                    consumers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::NNDMA) {
                    consumers.push_back(userOp);
                } else if (task.getTaskType() == VPUIP::TaskType::UPA) {
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

void FeasibleScheduleGenerator::compute_op_indegree(operation_in_degree_t& in_degree) {
    in_degree.clear();

    _func.walk([&](VPURT::TaskOp taskOp) {
        auto& block = taskOp.op().getBlocks().front();
        auto wrappedTaskOp = block.begin();
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
    std::cout << "The size of indegree table is " << in_degree.size() << std::endl;
}

bool FeasibleScheduleGenerator::doesOpRunOnNCE(mlir::Operation* op) {
    auto task = mlir::dyn_cast<VPURT::TaskOp>(op);
    if ((mlir::dyn_cast<VPURT::TaskOp>(op).getTaskType() ==  VPUIP::TaskType::NCE2) || (mlir::dyn_cast<VPURT::TaskOp>(op).getTaskType() == VPUIP::TaskType::NNDMA))
        return true;
    else
        return false;
}

unsigned FeasibleScheduleGenerator::countProducerConsumerTasks(mlir::Operation* op) {
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getTaskType() ==  VPUIP::TaskType::NCE2)
        return 5;
    if (mlir::dyn_cast<VPURT::TaskOp>(op).getTaskType() ==  VPUIP::TaskType::NNDMA)
        return 1;
    else
        exit(1);
}

void FeasibleScheduleGenerator::create_resource_utility_table_for_barrier_scheduling() {
    for (auto& op : in_degree_) {
        if (doesOpRunOnNCE(op.first)) {
            auto resource_utility = countProducerConsumerTasks(op.first);
            // resource utility //
            Logger::global().error("Operation: {0} uses {1} slots", getUniqueID(op.first), resource_utility);
            resource_utility_map_.insert(std::make_pair(op.first, resource_utility));
        } else  // UPA tasks
        {
            // resource utility is 0 //
            Logger::global().error("Operation: {0} uses 0 slots", getUniqueID(op.first));
            resource_utility_map_.insert(std::make_pair(op.first, 0));
        }
    }
}

bool FeasibleScheduleGenerator::init(const resource_state_t& upper_bound) {
    Logger::global().error("**Initializing the feasible scheduler **");

    // printInfo(_func);
    assignUniqueIds(_func);
    processed_ops_.clear();
    resource_utility_map_.clear();

    getAllBarriersProducersAndConsumers();

    Logger::global().error("Initializing the resource upper state");
    init_resource_state(upper_bound);

    compute_op_indegree(in_degree_);

    create_resource_utility_table_for_barrier_scheduling();

    // collect the ones with zero-in degree into candidates //
    candidates_.clear();

    operation_in_degree_t::iterator itr = in_degree_.begin();
    operation_in_degree_t::iterator itr_end = in_degree_.end();

    while (itr != itr_end) {
        auto op = itr->first;
        if (in_degree_.find(op)->second == 0) {
            Logger::global().error("Adding op: {0} to candidate set", getUniqueID(op));
            add_to_candidate_set(op);
        }
        ++itr;
    }

    if (candidates_.empty()) {
        fprintf(stderr, "Feasible_Schedule_Generator: no operations with ZERO"
                        " in-degree means there must be a cycle in the input");
        return false;
    }
    compute_operation_priorities();

    return next_schedulable_operation();
}
