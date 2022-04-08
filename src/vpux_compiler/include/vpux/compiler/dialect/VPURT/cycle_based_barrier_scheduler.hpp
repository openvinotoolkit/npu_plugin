//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#pragma once

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPURT/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/cycle_based_barrier_resource_state.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"

#include <llvm/ADT/BitVector.h>

namespace vpux {

namespace VPURT {

class CycleBasedBarrierScheduler final {
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

    struct startCycleTaskComparator {
        bool operator()(VPURT::TaskOp& op1, VPURT::TaskOp& op2) const;
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

    using activeBarrierMapType = std::unordered_map<mlir::Operation*, barrierInfo>;
    using scheduleHeapType = std::vector<HeapElement>;
    using barrierResourceUtilityMapType = std::unordered_map<mlir::Operation*, size_t>;
    using barrierWaitMapType = SmallVector<llvm::BitVector>;
    using barrierUpdateMapType = SmallVector<llvm::BitVector>;
    using taskOpWaitMapType = SmallVector<llvm::BitVector>;
    using taskOpUpdateMapType = SmallVector<llvm::BitVector>;
    using taskAndCyclePair = std::pair<vpux::VPURT::TaskOp, size_t>;

    CycleBasedBarrierScheduler(mlir::FuncOp func, Logger log);
    void init(size_t numberOfBarriers, size_t maxProducersPerBarrier);
    void clearTemporaryAttributes();
    bool generateScheduleWithBarriers();
    bool performRuntimeSimulation();

private:
    void assignTaskUniqueIds();
    void getPerTaskStartAndEndCycleTime();
    void optimizeIRDependency();
    void createTaskBarrierResourceUtilityTable();
    void initializeBarrierAssociationTable();
    void pushToScheduleTimeHeap(const HeapElement& elem);
    void insertBarriersinIR();
    void populateScheduledTasks(mlir::Operation* scheduledOp);

    void removeRedundantDependencies();
    bool canBarriersBeMerged(size_t barrier1, size_t barrier2);
    void removeRedundantBarriers(bool optimizeIRDependency);
    void reorderIR();
    void removeVirtualBarriers();

    bool performCycleBasedSchedulingTaskLoop();
    bool isBarrierResourceAvailable(const size_t demand, mlir::Operation* task, bool scheduledTasksAllFinished);
    bool scheduleTask(mlir::Operation* op, const size_t demand);
    bool unScheduleTask(mlir::Operation* op);
    bool doesPathExist(int64_t a, int64_t b, bool checkConsumer);
    Optional<taskAndCyclePair> updateCycleStartTimeDueToNativeExecutorDependency(
            vpux::VPURT::TaskOp task, size_t newStartCycle, SmallVector<vpux::VPURT::TaskOp>& orderedTasksByCycleStart);
    void updateCycleStartTime(vpux::VPURT::TaskOp srcTaskOp, size_t srcNewStartCycle);
    void optimizeDependency();
    void updateDependency();
    bool attemptToScheduleTaskWithAvailableBarrier(VPURT::TaskOp taskOp);

    const CycleBasedBarrierResourceState& barrierResourceState() const;
    size_t countProducerTasksToBarrier(VPURT::TaskOp op);
    static size_t getUniqueID(mlir::Operation* op);
    const barrierInfo& getBarrierInfo(mlir::Operation* op) const;

    // The number of available barriers
    size_t _barrierCount;
    // The number of available producers slots per barrier
    size_t _slotsPerBarrier;
    // The current barrier resource utilization by the schedule i.e active barriers and the number of producers
    CycleBasedBarrierResourceState _barrierResourceState;
    // Heap with operation end time
    scheduleHeapType _heap;
    // The current time of the list schedule
    size_t _currentTime;
    // The number of producer slots in a barrier that a task requires
    barrierResourceUtilityMapType _barrierResourceUtilizationMap;
    // The barrier information for each task
    activeBarrierMapType _barrierMap;
    // Stores every barrier's associated update and wait operations
    barrierWaitMapType _configureBarrierOpWaitTask;
    barrierUpdateMapType _configureBarrierOpUpdateTask;
    // Stores every task's associated update and wait barriers
    taskOpWaitMapType _configureTaskOpWaitBarrier;
    taskOpUpdateMapType _configureTaskOpUpdateBarrier;
    // The consumer tasks per task from original dependency
    std::map<mlir::Operation*, std::set<mlir::Operation*>> _taskConsumerMapOriginal;
    Logger _log;
    mlir::FuncOp _func;
    // Enable DMA related barrier optimization
    bool _enableDMAOptimization;
    // The number of execute tasks
    size_t _taskCount;
    // The vector of ordered barriers
    SmallVector<VPURT::DeclareVirtualBarrierOp> _orderedBarrier;
    // The vector of ordered execute tasks by uniqueID
    SmallVector<VPURT::TaskOp> _orderedTasks;
    // The vector of ordered execute tasks by scheduling number, which is used to reorder tasks' position in IR
    SmallVector<size_t> _schedulingOrder;
    // A set of ordered time stamp. It contains the cycle time which is necessary for scheduler to check. So it's
    // initialized with the start time from memory scheduler.
    std::set<size_t> _orderedTimeStamp;
    // A vector of cycle start time. Each index corresponds to the operation ID
    SmallVector<size_t> _operationStartCycle;
    // A vector of cycle end time. Each index corresponds to the operation ID
    SmallVector<size_t> _operationEndCycle;
    // A map of ordered execute tasks by cycle start
    std::map<TaskQueueType, SmallVector<VPURT::TaskOp>> _orderedTasksByCycleStart;
    DenseMap<VPURT::TaskOp, bool> _taskScheduleStatus;
    // A vector of physical barrier ID assigned by scheduler
    SmallVector<size_t> _physicalID;
    // The look up table for path existing. The defination of std::tuple<int64_t, int64_t, bool> is <source task ID,
    // target task ID, check path for barrier's consumer or producer>.
    // If checking path for barrier's producer, a True value means target task ends after source task ends.
    // If checking path for barrier's consumer, a True value means target task starts after source task starts.
    std::map<std::tuple<int64_t, int64_t, bool>, bool> _pathLookUpTable;
    // The vector of scheduled tasks in each loop
    SmallVector<size_t> _schedulingTasksInEachLoop;
};

}  // namespace VPURT
}  // namespace vpux
