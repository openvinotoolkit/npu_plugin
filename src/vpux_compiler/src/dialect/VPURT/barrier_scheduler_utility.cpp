//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"

using namespace vpux::VPURT;

bool BarrierScheduler::scheduleNumberTaskComparator::operator()(mlir::Operation* op1, mlir::Operation* op2) const {
    int64_t schedulingNumber1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(schedulingNumberAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t schedulingNumber2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(schedulingNumberAttrName).cast<mlir::IntegerAttr>().getInt());

    return schedulingNumber1 < schedulingNumber2;
}

bool BarrierScheduler::uniqueBarrierIDTaskComparator::operator()(mlir::Operation* op1, mlir::Operation* op2) const {
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

bool BarrierScheduler::uniqueTaskIDTaskComparator::operator()(mlir::Operation* op1, mlir::Operation* op2) const {
    int64_t uniqueId1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t uniqueId2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
    return uniqueId1 < uniqueId2;
}

bool BarrierScheduler::barrierTransitionStructure::uniqueIDTaskComparator::operator()(mlir::Operation* op1,
                                                                                      mlir::Operation* op2) const {
    int64_t uniqueId1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t uniqueId2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());

    return uniqueId1 < uniqueId2;
}
