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

namespace vpux {

namespace VPURT {

constexpr llvm::StringLiteral schedulingNumberAttrName = "SchedulingNumber";
constexpr llvm::StringLiteral uniqueIdAttrName = "uniqueId";
constexpr llvm::StringLiteral virtualIdAttrName = "VPURT.virtualId";
class FeasibleBarrierScheduler final {
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
                                                              ->getAttr(virtualIdAttrName)
                                                              .cast<mlir::IntegerAttr>()
                                                              .getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op2)
                                                              ->getAttr(virtualIdAttrName)
                                                              .cast<mlir::IntegerAttr>()
                                                              .getInt());

            return uniqueId1 < uniqueId2;
        }
    };

    using schedule_time_t = size_t;
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

    struct barrier_info_t {
    barrier_info_t(size_t bindex = 0UL, size_t slot_count = 0UL): bindex_(bindex), slot_count_(slot_count) {
    }
    size_t bindex_;
    size_t slot_count_;
};
    class barrierTransitionStructure {
    public:
        barrierTransitionStructure(FeasibleBarrierScheduler& feasibleBarrierScheduler,
                                   schedule_time_t time = std::numeric_limits<schedule_time_t>::max());

        void init();
        bool processNextScheduledTask(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        void closeBarrierProducerList();
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
        void maintainInvariantTemporalChange(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        inline void processCurrentBarrierProducerListCloseEvent(mlir::Operation* bop_curr,
                                                                      mlir::Operation* bop_prev);
        void addScheduledOpToProducerList(const ScheduledOpInfo& sinfo);
        mlir::Operation* createNewBarrierTask(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);

        
        // Outer class
        FeasibleBarrierScheduler& _feasibleBarrierScheduler;
        schedule_time_t time_;
        mlir::Operation* curr_barrier_task_;
        mlir::Operation* prev_barrier_task_;
        producers_t producers_;
    };

    using operation_t = mlir::Operation*;
    using resource_t = size_t;
    using active_barrier_map_t = std::unordered_map<operation_t, barrier_info_t>;
    using barrierAssociationTable = std::unordered_map<size_t, barrierTransitionStructure>;
    using delay_t = size_t;
    using schedulable_ops_t = std::list<mlir::Operation*>;
    using schedulable_ops_iterator_t = typename schedulable_ops_t::iterator;
    using processed_ops_t = std::set<mlir::Operation*>;
    using schedule_heap_t = std::vector<HeapElement>;
    using operation_in_degree_t = std::map<mlir::Operation*, size_t>;
    using operation_out_degree_t = std::map<mlir::Operation*, size_t>;
    using priority_map_t = std::map<mlir::Operation*, size_t>;
    using resource_utility_map_t = std::unordered_map<mlir::Operation*, unsigned>;
    

    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log);
    void initializeBarrierResourceState(size_t numberOfBarriers, size_t maxProducersPerBarrier);
    bool isBarrierResourceAvailable(const resource_t& demand);
    bool scheduleOperation(mlir::Operation*& op, resource_t demand);
    bool unScheduleOperation(mlir::Operation*& op);
    void addTaskToCandidateSet(mlir::Operation* op);
    void computeTaskPriorities();
    void createTaskBarrierResourceUtilityTable();
    void addOutGoingOperationsToCandidateList(mlir::Operation* op);
    void assignTaskUniqueIds();
    void pushToHeap(const HeapElement& elem);
    HeapElement popFromHeap();
    void saveOriginalIRDependency();

    bool scheduleOperations();
    bool schedule(size_t numberOfBarriers, size_t maxProducersPerBarrier);
    void init();
    bool doesOpRunOnNCE(mlir::Operation* op);

    size_t currentTime() const;
    const BarrierResourceState& barrierResourceState() const;
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
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                     operation_comparator_t>& barrierOpUpdateWaitMap,
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                     task_operation_comparator_by_schedule_time_t>& taskOpUpdateWaitMap);
    void removeRedundantDependency();
    void removeRedundantBarrier();
    bool isPathExist(mlir::Operation* a, mlir::Operation* b);
    void reorderIR();
    bool performRuntimeSimulation();
    void cleanUpVirtualBarriers();
    const barrier_info_t& get_barrier_info(const operation_t& op) const;

private:
    // The number of available barriers
    size_t _barrierCount;
    // The number of available producers slots per barrier
    size_t _slotsPerBarrier;
    // The current barrier resource utilization by the schedule i.e active barriers and the number of producers per
    // barrier
    //resource_state_t _resourceState;
    BarrierResourceState _barrierResourceState;
    // The in-degree per task
    operation_in_degree_t _inDegree;
    // Heap with operation end time
    schedule_heap_t _heap;
    // The current time of the list schedule
    schedule_time_t _currentTime;
    // Tasks that can be scheduled i.e. task with in-degree zero
    schedulable_ops_t _scheduleableCandidates;
    // Current task that is being scheduled
    mlir::Operation* _schedulableTask;
    // Tasks that have been scheduled
    processed_ops_t _processedTasks;
    // The scheduling priority of tasks
    priority_map_t _priority;
    // container for the schedule output
    SmallVector<ScheduledOpInfo> _scheduledTasks;
    // A map of physical barriers and their transition structure
    barrierAssociationTable _barrierAssociationTable;
    // The number of producer slots in a barrier that a task requires
    resource_utility_map_t _resourceUtilityMap;
    // The output tasks in the IR
    std::set<mlir::Operation*> _outputOps;

    active_barrier_map_t _barrierMap;
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;

    /*Stores every barrier's associated update and wait operations*/
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
             operation_comparator_t>
            configureBarrierOpUpdateWaitMap;  // update,wait
    // Stores every task's associated update and wait barriers
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
             task_operation_comparator_by_schedule_time_t>
            configureTaskOpUpdateWaitMap;
    // The in-degree per task from original dependency
    operation_in_degree_t _inDegreeBackUp;
    // The consumer tasks per task from original dependency
    std::map<mlir::Operation*, SmallVector<mlir::Operation*>> _taskConsumerMapBackUp;
    // The backup of output tasks in the IR
    std::set<mlir::Operation*> _outputOpsBackUp;
};

}  // namespace VPURT
}  // namespace vpux
