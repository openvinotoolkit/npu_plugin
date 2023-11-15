//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallSet.h>

namespace vpux {

class BarrierInfo final {
public:
    // TaskSet is used to store barrier's producer/consumer task op index as well as task op's
    // wait/update barrier index, which is supposed to have better performance than BitVector when the data size is
    // small.
    using TaskSet = llvm::SmallSet<size_t, 16>;
    explicit BarrierInfo(mlir::func::FuncOp func);

public:
    void updateIR();
    void orderBarriers();
    void clearAttributes();
    TaskSet getWaitBarriers(size_t taskInd);
    TaskSet getUpdateBarriers(size_t taskInd);
    uint32_t getIndex(VPURT::TaskOp taskOp) const;
    uint32_t getIndex(VPURT::DeclareVirtualBarrierOp barrierOp) const;
    VPURT::TaskOp getTaskOpAtIndex(size_t opIdx) const;
    VPURT::DeclareVirtualBarrierOp getBarrierOpAtIndex(size_t opIdx) const;

private:
    void addTaskOp(VPURT::TaskOp taskOp);
    void buildBarrierMaps(mlir::func::FuncOp func);
    void setWaitBarriers(size_t taskIdn, const TaskSet& barriers);
    void setUpdateBarriers(size_t taskIdn, const TaskSet& barriers);
    void resizeBitMap(SmallVector<llvm::BitVector>& bitMap, size_t length, uint32_t bits);
    bool producersControllsAllConsumers(const TaskSet& origProducers, const TaskSet& newConsumers,
                                        const TaskSet& origConsumers, ArrayRef<TaskSet> origWaitBarriersMap);
    bool inImplicitQueueTypeDependencyList(const TaskSet& taskList);

public:
    void logBarrierInfo();
    void optimizeBarriers();
    void buildTaskControllMap(bool considerTaskFifoDependency = true);
    size_t getNumOfTasks() const;
    size_t getNumOfVirtualBarriers() const;
    size_t getNumOfSlotsUsed(VPURT::TaskOp op) const;
    void resetBarrier(VPURT::DeclareVirtualBarrierOp barrierOp);
    void resetBarrier(size_t barrierInd);
    void addNewBarrier(VPURT::DeclareVirtualBarrierOp barrierOp);
    bool controlPathExistsBetween(size_t taskAInd, size_t taskBInd, bool bidirection = true) const;
    size_t getProducerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp);
    size_t getConsumerSlotCount(VPURT::DeclareVirtualBarrierOp barrierOp);
    void addProducer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd);
    void addProducers(size_t barrierInd, const TaskSet& taskInds);
    void addConsumer(VPURT::DeclareVirtualBarrierOp barrierOp, size_t taskInd);
    void addConsumers(size_t barrierInd, const TaskSet& taskInds);
    void removeProducer(size_t taskInd, VPURT::DeclareVirtualBarrierOp barrierOp);
    void removeConsumer(size_t taskInd, VPURT::DeclareVirtualBarrierOp barrierOp);
    void removeProducers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds);
    void removeConsumers(VPURT::DeclareVirtualBarrierOp barrierOp, const TaskSet& taskInds);
    TaskSet getBarrierProducers(VPURT::DeclareVirtualBarrierOp barrierOp);
    TaskSet getBarrierConsumers(VPURT::DeclareVirtualBarrierOp barrierOp);
    TaskSet getBarrierProducers(size_t barrierIdn);
    TaskSet getBarrierConsumers(size_t barrierIdn);
    SmallVector<TaskSet> createLegalVariantBatches(const TaskSet& tasks, size_t availableSlots);
    Optional<VPURT::TaskQueueType> haveSameImplicitDependencyTaskQueueType(const TaskSet& taskInds);
    bool canBarriersBeMerged(const TaskSet& barrierProducersA, const TaskSet& barrierConsumersA,
                             const TaskSet& barrierProducersB, const TaskSet& barrierConsumersB,
                             ArrayRef<TaskSet> origWaitBarriersMap);
    SmallVector<TaskSet> getWaitBarriersMap();

private:
    Logger _log;

    mlir::StringAttr _taskIndexAttrName;
    mlir::StringAttr _barrierIndexAttrName;

    SmallVector<VPURT::TaskOp> _allTaskOps;
    SmallVector<VPURT::DeclareVirtualBarrierOp> _allBarrierOps;

    // Note:
    //  - task produces its update barriers
    //  - task consumes its wait barriers

    // indexOf(VPURT::DeclareVirtualBarrierOp) 'is produced by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierProducerMap;
    // indexOf(VPURT::DeclareVirtualBarrierOp) 'is consumed by' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<TaskSet> _barrierConsumerMap;

    // indexOf(VPURT::TaskOp) 'waits for' [ indexOf(VPURT::DeclareVirtualBarrierOp)... ].
    SmallVector<TaskSet> _taskWaitBarriers;
    // indexOf(VPURT::TaskOp) 'updates' [ indexOf(VPURT::DeclareVirtualBarrierOp)... ].
    SmallVector<TaskSet> _taskUpdateBarriers;

    // indexOf(VPURT::TaskOp) 'controlls' [ indexOf(VPURT::TaskOp)... ].
    SmallVector<llvm::BitVector> _taskControllMap;

    // indexOf(VPURT::TaskQueueType) 'contains' [ indexOf(VPURT::TaskOp)... ].
    std::map<VPURT::TaskQueueType, llvm::BitVector> _taskQueueTypeMap;
};

}  // namespace vpux
