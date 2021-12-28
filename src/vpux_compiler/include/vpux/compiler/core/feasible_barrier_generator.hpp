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
#pragma once

#include "vpux/compiler/dialect/VPURT/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/core/barrier_resource_state.hpp"
#include "vpux/compiler/core/op_resource_state.hpp"

namespace vpux {

struct opResourceState;
class FeasibleBarrierScheduler {
public:
    struct task_operation_comparator_by_schedule_time_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t schedulingNumber1 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());
            int64_t schedulingNumber2 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());

            return schedulingNumber1 < schedulingNumber2;
        }
    };
    struct operation_comparator_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t uniqueId1 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op1)
                                                              ->getAttr("id")
                                                              .cast<mlir::IntegerAttr>()
                                                              .getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op2)
                                                              ->getAttr("id")
                                                              .cast<mlir::IntegerAttr>()
                                                              .getInt());

            return uniqueId1 < uniqueId2;
        }
    };
    struct task_operation_comparator_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t uniqueId1 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());

            return uniqueId1 < uniqueId2;
        }
    };

    struct HeapElement {
        HeapElement(mlir::Operation* op = NULL, schedule_time_t t = 0UL): op_(op), time_(t) {
        }

        mlir::Operation* op_;
        schedule_time_t time_;
    };

    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            return a.time_ > b.time_;
        }
    };

    struct ScheduledOpInfo {
        schedule_time_t schedule_time_;
        mlir::Operation* op_;
        size_t barrier_index_;
        size_t slot_count_;
    };

    struct barrierInfo {
        barrierInfo(size_t bindex = 0UL, size_t slot_count = 0UL): bindex_(bindex), slot_count_(slot_count) {
        }
        size_t bindex_;
        size_t slot_count_;
    }; /*struct barrierInfo*/

    class barrierTransitionStructure {
    public:
        barrierTransitionStructure(mlir::FuncOp func, FeasibleBarrierScheduler& feasibleBarrierScheduler,
                                   schedule_time_t time = std::numeric_limits<schedule_time_t>::max());

        void init();
        bool process_next_scheduled_op(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        void close_barrier_producer_list();
        struct operation_comparator_t {
            bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
                int64_t uniqueId1 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::TaskOp>(op1)
                                                                  ->getAttr(uniqueIdAttrName)
                                                                  .cast<mlir::IntegerAttr>()
                                                                  .getInt());
                int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::TaskOp>(op2)
                                                                  ->getAttr(uniqueIdAttrName)
                                                                  .cast<mlir::IntegerAttr>()
                                                                  .getInt());

                return uniqueId1 < uniqueId2;
            }
        };

        using producers_t = std::set<mlir::Operation*, operation_comparator_t>;
        using producer_iterator_t = typename producers_t::const_iterator;

    private:
        void maintain_invariant_temporal_change(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        inline void process_current_barrier_producer_list_close_event(mlir::Operation* bop_curr,
                                                                      mlir::Operation* bop_prev);
        void add_scheduled_op_to_producer_list(const ScheduledOpInfo& sinfo);
        mlir::Operation* create_new_barrier_task(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);

        mlir::FuncOp _func;
        // Outer class
        FeasibleBarrierScheduler& feasibleBarrierScheduler_;
        schedule_time_t time_;
        mlir::Operation* curr_barrier_task_;
        mlir::Operation* prev_barrier_task_;
        producers_t producers_;
    };

    typedef mlir::Operation const* operation_t;
    typedef std::unordered_map<operation_t, barrierInfo> active_barrier_map_t;
    typedef size_t resource_t;
    typedef std::unordered_map<size_t, barrierTransitionStructure> barrierAssociationTable;
    using delay_t = size_t;
    using schedulable_ops_t = std::list<mlir::Operation*>;
    using schedulable_ops_iterator_t = typename schedulable_ops_t::iterator;
    using processed_ops_t = std::set<mlir::Operation*>;
    using schedule_heap_t = std::vector<HeapElement>;
    using operation_in_degree_t = std::map<mlir::Operation*, size_t, task_operation_comparator_t>;
    using operation_out_degree_t = std::map<mlir::Operation*, size_t, task_operation_comparator_t>;
    using priority_map_t = std::map<mlir::Operation*, size_t, task_operation_comparator_t>;
    using resource_utility_map_t = std::unordered_map<mlir::Operation*, unsigned>;
    using schedule_time_t = size_t;

    struct opResourceState {
        opResourceState(size_t n = 0UL, size_t m = 0UL)
                : barrier_map_(), state_(), barrier_count_(n), slots_per_barrier_(m) {
            Logger::global().error(
                    "Initializing op_resource_state in Barrier_Schedule_Generator with barrier count {0} "
                    "slots_per_barrie {1}",
                    barrier_count_, slots_per_barrier_);
        }

        void init(const opResourceState& other) {
            barrier_map_.clear();
            barrier_count_ = other.barrier_count_;
            slots_per_barrier_ = other.slots_per_barrier_;
            state_.init(barrier_count_, slots_per_barrier_);
        }

        bool is_resource_available(const resource_t& demand) const {
            Logger::global().error("Looking for a barrier with free slots");
            return state_.has_barrier_with_slots(demand);
        }

        bool schedule_operation(const operation_t& op, resource_t& demand) {
            Logger::global().error("Scheduling an operation");
            assert(is_resource_available(demand));
            if (barrier_map_.find(op) != barrier_map_.end()) {
                return false;
            }
            size_t bid = state_.assign_slots(demand);
            barrier_map_.insert(std::make_pair(op, barrierInfo(bid, demand)));
            return true;
        }

        bool unschedule_operation(const operation_t& op) {
            auto itr = barrier_map_.find(op);
            if (itr == barrier_map_.end()) {
                return false;
            }
            const barrierInfo& binfo = itr->second;
            bool ret = state_.unassign_slots(binfo.bindex_, binfo.slot_count_);
            assert(ret);
            (void)ret;
            barrier_map_.erase(itr);
            return true;
        }

        const barrierInfo& get_barrier_info(const operation_t& op) const {
            auto itr = barrier_map_.find(op);

            assert(itr != barrier_map_.end());
            return itr->second;
        }

        active_barrier_map_t barrier_map_;
        BarrierResourceState state_;
        size_t barrier_count_;
        size_t slots_per_barrier_;
    }; /*struct opResourceState*/

    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log);

    void operator++();
    void getBarriersProducersAndConsumers();
    void initResourceState(size_t numberOfBarriers, size_t maxProducersPerBarrier);
    bool isResourceAvailable(const resource_t& demand);
    bool scheduleOperation(mlir::Operation*& op, resource_t demand);
    bool unScheduleOperation(mlir::Operation*& op);
    void computeOpIndegree(operation_in_degree_t& in_degree);
    void computeOpOutdegree(operation_out_degree_t& out_degree);
    void addToCandidateSet(mlir::Operation* op);
    void computeOperationPriorities();
    void createOperationResourceUtilityTable();
    void addOutGoingOperationsToCandidateList(mlir::Operation* op);
    void assignUniqueIds();
    void pushToHeap(const HeapElement& elem);
    HeapElement popFromHeap();
    void saveOriginalDependency();

    bool operator==(const FeasibleBarrierScheduler& o) const;
    bool reached_end() const;
    bool scheduleOperations();
    bool schedule(size_t numberOfBarriers, size_t maxProducersPerBarrier);
    void init();
    bool doesOpRunOnNCE(mlir::Operation* op);

    mlir::Operation*& operator*();
    size_t currentTime() const;
    const resource_state_t& resourceState() const;
    bool isValidOp(schedulable_ops_iterator_t itr) const;
    schedulable_ops_iterator_t findSchedulableOp();
    unsigned countProducerConsumerTasks(mlir::Operation* op);
    SmallVector<mlir::Operation*> getConsumerOps(mlir::Operation* op);
    static std::string printOpType(VPURT::TaskOp taskOp);
    static mlir::IntegerAttr getUniqueID(mlir::Operation* op);
    void insertBarriersinIR();
    void populateScheduledOps(mlir::Operation* scheduledOp);
    void removeRedundantWaitBarriers();
    void removeRedundantDependencies();
    void initializeBarrierAssociationTable();
    void getTaskOpUpdateWaitMap(
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                    barrierOpUpdateWaitMap,
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                    taskOpUpdateWaitMap);
    void getTaskOpUpdateWaitMap(
            std::map<mlir::Operation*,
                     std::pair<std::set<mlir::Operation*, task_operation_comparator_t>,
                               std::set<mlir::Operation*, task_operation_comparator_t>>,
                     operation_comparator_t>& barrierOpUpdateWaitMap,
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                     task_operation_comparator_by_schedule_time_t>& taskOpUpdateWaitMap);
    void removeRedundantDependency();
    void removeRedundantBarrier();
    bool isPathExist(mlir::Operation* a, mlir::Operation* b);
    void reorderIR();
    bool performRuntimeSimulation();
    void cleanUpVirtualBarriers();

protected:
    size_t _barrierCount;
    size_t _slotsPerBarrier;
    resource_state_t _resource_state;
    operation_in_degree_t _in_degree;
    operation_out_degree_t _out_degree;
    schedule_heap_t _heap;
    schedule_time_t _current_time;
    schedulable_ops_t _candidates;
    MinHeapOrdering _heap_ordering;
    mlir::Operation* _schedulable_op;
    processed_ops_t _processed_ops;
    priority_map_t _priority;
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;

    resource_utility_map_t _resource_utility_map;
    SmallVector<IERT::LayerOpInterface> _allTaskOps;
    SmallVector<VPURT::DeclareVirtualBarrierOp> _allBarrierOps;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierProducersMap;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierConsumersMap;
    std::set<mlir::Operation*> _outputOps;

    const resource_state_t _startState;
    // container for the schedule output
    SmallVector<ScheduledOpInfo> _scheduledOps;
    barrierAssociationTable _barrierAssociationTable;

    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            _barrierOpUpdateWaitMap;
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            _taskOpUpdateWaitMap{};

    /*Stores every barrier's associated update and wait operations*/
    std::map<mlir::Operation*,
             std::pair<std::set<mlir::Operation*, task_operation_comparator_t>,
                       std::set<mlir::Operation*, task_operation_comparator_t>>,
             operation_comparator_t>
            configureBarrierOpUpdateWaitMap;  // update,wait

    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
             task_operation_comparator_by_schedule_time_t>
            configureTaskOpUpdateWaitMap;  // update,wait

    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            configureTaskOpUpdateWaitMapBackUp;  // update,wait

    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            configureBarrierOpUpdateWaitMapBackUp;  // update,wait
};

}  // namespace vpux
