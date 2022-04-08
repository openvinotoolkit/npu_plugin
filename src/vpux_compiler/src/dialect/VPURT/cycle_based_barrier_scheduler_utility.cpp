//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/cycle_based_barrier_scheduler.hpp"

using namespace vpux::VPURT;

bool CycleBasedBarrierScheduler::scheduleNumberTaskComparator::operator()(mlir::Operation* op1,
                                                                          mlir::Operation* op2) const {
    int64_t schedulingNumber1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(schedulingNumberAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t schedulingNumber2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(schedulingNumberAttrName).cast<mlir::IntegerAttr>().getInt());

    return schedulingNumber1 < schedulingNumber2;
}

bool CycleBasedBarrierScheduler::uniqueBarrierIDTaskComparator::operator()(mlir::Operation* op1,
                                                                           mlir::Operation* op2) const {
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

bool CycleBasedBarrierScheduler::uniqueTaskIDTaskComparator::operator()(mlir::Operation* op1,
                                                                        mlir::Operation* op2) const {
    int64_t uniqueId1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t uniqueId2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
    return uniqueId1 < uniqueId2;
}

bool CycleBasedBarrierScheduler::startCycleTaskComparator::operator()(VPURT::TaskOp& op1, VPURT::TaskOp& op2) const {
    int64_t startCycle1 = checked_cast<int64_t>(op1->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getInt());
    int64_t startCycle2 = checked_cast<int64_t>(op2->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getInt());

    return startCycle1 < startCycle2;
}
