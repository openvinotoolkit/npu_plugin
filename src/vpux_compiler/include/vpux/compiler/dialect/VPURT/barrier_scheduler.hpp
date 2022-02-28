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

#include "vpux/compiler/dialect/VPURT/barrier_resource_state.hpp"
#include "vpux/compiler/dialect/VPURT/barrier_simulator.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include <llvm/ADT/BitVector.h>

namespace vpux {

namespace VPURT {

constexpr llvm::StringLiteral schedulingNumberAttrName = "SchedulingNumber";
constexpr llvm::StringLiteral uniqueIdAttrName = "uniqueId";
constexpr llvm::StringLiteral virtualIdAttrName = "VPURT.virtualId";

class BarrierScheduler final {
public:
    struct scheduleNumberTaskComparator {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const;
    };

    struct uniqueBarrierIDTaskComparator {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const;
    };

    struct uniqueTaskIDTaskComparator {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const;
    };

    struct HeapElement {
        HeapElement(mlir::Operation* op = nullptr, size_t t = 0UL): _op(op), _time(t) {
        }
        mlir::Operation* _op;
        size_t _time;
    };

    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            return a._time > b._time;
        }
    };

    struct ScheduledOpInfo {
        size_t _scheduleTime;
        mlir::Operation* _op;
        size_t _barrierIndex;
        size_t _producerSlotCount;
    };

    struct barrierInfo {
        barrierInfo(size_t barrierIndex = 0UL, size_t slotCount = 0UL)
                : _barrierIndex(barrierIndex), _producerSlotCount(slotCount) {
        }
        size_t _barrierIndex;
        size_t _producerSlotCount;
    };

    class barrierTransitionStructure {
    public:
        barrierTransitionStructure(BarrierScheduler& feasibleBarrierScheduler, size_t taskCount,
                                   size_t time = std::numeric_limits<size_t>::max());

        void init();
        bool processNextScheduledTask(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        void closeBarrierProducerList();
        struct uniqueIDTaskComparator {
            bool operator()(mlir::Operation* op1, mlir::Operation* op2) const;
        };

        using producersType = std::set<mlir::Operation*, uniqueIDTaskComparator>;
        using producerIteratorType = typename producersType::const_iterator;

    private:
        void maintainInvariantTemporalChange(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        inline void processCurrentBarrierProducerListCloseEvent(mlir::Operation* currentBarrier,
                                                                mlir::Operation* previousBarrier);
        void addScheduledTaskToProducerList(const ScheduledOpInfo& sinfo);
        mlir::Operation* createNewBarrierTask(const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder);
        size_t getTaskUniqueID(mlir::Operation* task);
        size_t getBarrierUniqueID(mlir::Operation* barrier);

        BarrierScheduler& _feasibleBarrierScheduler;
        size_t _taskCount;
        size_t _time;
        mlir::Operation* _currentBarrierTask = nullptr;
        mlir::Operation* _previousBarrierTask = nullptr;
        producersType _producers;
    };

    using schedulableOpsType = std::list<mlir::Operation*>;
    using schedulableTasksIteratorType = typename schedulableOpsType::iterator;
    using activeBarrierMapType = std::unordered_map<mlir::Operation*, barrierInfo>;
    using barrierAssociationTableType = std::unordered_map<size_t, barrierTransitionStructure>;
    using processedOpsType = std::set<mlir::Operation*>;
    using scheduleHeapType = std::vector<HeapElement>;
    using operationInDegreeType = std::map<mlir::Operation*, size_t, uniqueTaskIDTaskComparator>;
    using priorityMapType = std::map<mlir::Operation*, size_t>;
    using barrierResourceUtilityMapType = std::unordered_map<mlir::Operation*, size_t>;
    using barrierWaitMapType = SmallVector<llvm::BitVector>;
    using barrierUpdateMapType = SmallVector<llvm::BitVector>;
    using taskOpWaitMapType = SmallVector<llvm::BitVector>;
    using taskOpUpdateMapType = SmallVector<llvm::BitVector>;

    BarrierScheduler(mlir::FuncOp func, Logger log);
    void init();
    void clearTemporaryAttributes();
    bool generateScheduleWithBarriers(const size_t numberOfBarriers, const size_t maxProducersPerBarrier);
    bool performRuntimeSimulation();

private:
    void assignTaskUniqueIds();
    void saveOriginalIRDependency();
    void assignTaskPriorities();
    void createTaskBarrierResourceUtilityTable();
    void initializeBarrierAssociationTable();
    void initializeBarrierResourceState(const size_t numberOfBarriers, const size_t maxProducersPerBarrier);
    void addTaskToCandidateSet(mlir::Operation* op);
    void addOutGoingOperationsToCandidateList(mlir::Operation* op);
    void pushToScheduleTimeHeap(const HeapElement& elem);
    void insertBarriersinIR();
    void populateScheduledTasks(mlir::Operation* scheduledOp);
    void removeRedundantWaitBarriers();
    void populateTasksUpdateWaitBarrierMap(barrierWaitMapType& barrierOpWaitMap,
                                           barrierUpdateMapType& barrierOpUpdateMap, taskOpWaitMapType& taskOpWaitMap,
                                           taskOpUpdateMapType& taskOpUpdateMap);
    void removeRedundantDependencies();
    void removeRedundantBarriers();
    void reorderIR();
    void removeVirtualBarriers();

    bool performSchedulingTaskLoop();
    bool isBarrierResourceAvailable(const size_t demand);
    bool scheduleTask(mlir::Operation* op, const size_t demand);
    bool unScheduleTask(mlir::Operation* op);
    bool isTaskInSchedulableCandidates(schedulableTasksIteratorType itr) const;
    bool doesPathExist(int64_t a, int64_t b);

    HeapElement popFromHeap();
    const BarrierResourceState& barrierResourceState() const;
    schedulableTasksIteratorType findSchedulableTask();
    size_t countProducerTasksToBarrier(mlir::Operation* op);
    SmallVector<mlir::Operation*> getConsumerOps(mlir::Operation* op);
    static mlir::IntegerAttr getUniqueID(mlir::Operation* op);
    const barrierInfo& getBarrierInfo(mlir::Operation* op) const;

    // The number of available barriers
    size_t _barrierCount;
    // The number of available producers slots per barrier
    size_t _slotsPerBarrier;
    // The current barrier resource utilization by the schedule i.e active barriers and the number of producers
    BarrierResourceState _barrierResourceState;
    // The in-degree per task
    operationInDegreeType _inDegree;
    // The in-degree per task from original dependency
    operationInDegreeType _originalInDegree;
    // Heap with operation end time
    scheduleHeapType _heap;
    // The current time of the list schedule
    size_t _currentTime;
    // Tasks that can be scheduled i.e. task with in-degree zero
    schedulableOpsType _schedulableCandidates;
    // Tasks that have been scheduled
    processedOpsType _processedTasks;
    // The scheduling priority of tasks
    priorityMapType _priority;
    // container for the schedule output
    SmallVector<ScheduledOpInfo> _scheduledTasks;
    // A map of physical barriers and their transition structure
    barrierAssociationTableType _barrierAssociationTable;
    // The number of producer slots in a barrier that a task requires
    barrierResourceUtilityMapType _barrierResourceUtilizationMap;
    // The output tasks in the IR
    std::set<mlir::Operation*> _outputTasks;
    // The backup of output tasks in the IR
    std::set<mlir::Operation*> _originalOutputOps;
    // The barrier information for each task
    activeBarrierMapType _barrierMap;
    // Stores every barrier's associated update and wait operations
    barrierWaitMapType _configureBarrierOpWaitMap;
    barrierUpdateMapType _configureBarrierOpUpdateMap;
    // Stores every task's associated update and wait barriers
    taskOpWaitMapType _configureTaskOpWaitMap;
    taskOpUpdateMapType _configureTaskOpUpdateMap;
    // The consumer tasks per task from original dependency
    std::map<mlir::Operation*, SmallVector<mlir::Operation*>> _taskConsumerMapOriginal;
    Logger _log;
    mlir::FuncOp _func;
    // The number of execute tasks
    size_t _taskCount;
    // The vector of ordered barriers
    SmallVector<VPURT::DeclareVirtualBarrierOp> _orderedBarrier;
    // The vector of ordered execute tasks by uniqueID
    SmallVector<VPURT::TaskOp> _orderedTasks;
    // The vector of ordered execute tasks by scheduling number
    SmallVector<size_t> _schedulingOrder;
};

}  // namespace VPURT
}  // namespace vpux
