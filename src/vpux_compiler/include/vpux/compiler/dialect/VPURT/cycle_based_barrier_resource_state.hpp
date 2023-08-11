//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/BitVector.h>
namespace vpux {

namespace VPURT {

class CycleBasedBarrierResourceState final {
public:
    CycleBasedBarrierResourceState();

    // Container that describes for each barrier ID
    // (1) The barriers total allowable producer number i.e. 256
    // (2) The current available producer slots i.e (256 - used slots)
    // (3) If the barrier is currently in use i.e its total slots != current available slots
    struct availableSlotKey {
        availableSlotKey(size_t slots = size_t(0UL), size_t barrier = size_t(0UL))
                : _availableProducerSlots(slots), _totalProducerSlots(slots), _barrier(barrier) {
        }

        bool operator<(const availableSlotKey& o) const {
            return (o._availableProducerSlots != _availableProducerSlots)
                           ? (_availableProducerSlots < o._availableProducerSlots)
                           : (_barrier < o._barrier);
        }

        bool isBarrierInUse() const {
            return _totalProducerSlots > _availableProducerSlots;
        }

        size_t _availableProducerSlots;
        const size_t _totalProducerSlots;
        size_t _barrier;
    };  // struct availableSlotKey

    using availableProducerSlotsType = std::set<availableSlotKey>;
    using constAvailableslotsIteratorType = typename availableProducerSlotsType::const_iterator;
    using availableSlotsIteratorType = typename availableProducerSlotsType::iterator;
    using barrierReferenceType = std::vector<availableSlotsIteratorType>;

    void init(const size_t barrierCount, const size_t maximumProducerSlotCount, SmallVector<VPURT::TaskOp> orderedTasks,
              std::map<TaskQueueType, SmallVector<VPURT::TaskOp>> orderedTasksByCycleStart,
              std::map<mlir::Operation*, std::set<mlir::Operation*>> taskConsumerMapOriginal);
    bool createUnusedBarrierByAdjustingConsumer(SmallVector<llvm::BitVector>& configureBarrierOpWaitMap,
                                                SmallVector<llvm::BitVector>& configureBarrierOpUpdateMap,
                                                SmallVector<llvm::BitVector>& configureTaskOpWaitMap);
    constAvailableslotsIteratorType findUnusedBarrierWithAvailableSlots(size_t slotDemand);
    constAvailableslotsIteratorType findBarrierWithMinimumCycleDelay(
            size_t slotDemand, size_t latestWaitBarrier, mlir::Operation* producerTask,
            SmallVector<llvm::BitVector>& configureBarrierOpWaitMap,
            SmallVector<llvm::BitVector>& configureBarrierOpUpdateMap);
    size_t assignBarrierSlots(size_t slotDemand, mlir::Operation* producerTask, size_t latestWaitBarrier,
                              size_t& virtualId, SmallVector<llvm::BitVector>& configureBarrierOpWaitMap,
                              SmallVector<llvm::BitVector>& configureBarrierOpUpdateMap);
    bool assignBarrierSlots(size_t barrierId, size_t slotDemand);
    bool unassignBarrierSlots(size_t barrierId, size_t slotDemand, mlir::Operation* op);
    static size_t invalidBarrier();
    void update(size_t barrierId, size_t updatedAvailableProducerSlots);
    availableSlotsIteratorType update(availableSlotsIteratorType itr, size_t updatedAvailableProducerSlots);
    void updateBarrierConsumer(mlir::Operation* task, size_t barrierId);
    size_t getTaskUniqueID(mlir::Operation* task);
    vpux::VPURT::TaskOp findPreviousScheduledTask(vpux::VPURT::TaskOp task,
                                                  SmallVector<vpux::VPURT::TaskOp>& orderedTasksByCycleStart);
    size_t getTaskStartCycle(mlir::Operation* task);
    size_t getTaskEndCycle(mlir::Operation* task);

    // Stores an availableSlotKey struct for each barrier.
    // i.e. Information for each barrier, its total available slots (256) and the current free slots
    availableProducerSlotsType _globalAvailableProducerSlots;

private:
    // A vector of iterators to each entry in the _availableProducerSlots container
    barrierReferenceType _barrierReference;
    // A vector of producer list for each physical barrier
    SmallVector<std::set<mlir::Operation*>> _barrierProducers;
    // A vector of consumer list for each physical barrier
    SmallVector<std::set<mlir::Operation*>> _barrierConsumers;
    // A vector of virtual ID associated with each physical barrier
    SmallVector<int64_t> _physicalToVirtual;
    // The vector of ordered execute tasks by uniqueID
    SmallVector<VPURT::TaskOp> _orderedTasks;
    // A map of ordered execute tasks by cycle start
    std::map<TaskQueueType, SmallVector<VPURT::TaskOp>> _orderedTasksByCycleStart;
    // The consumer tasks per task from original dependency
    std::map<mlir::Operation*, std::set<mlir::Operation*>> _taskConsumerMapOriginal;
};

}  // namespace VPURT
}  // namespace vpux
