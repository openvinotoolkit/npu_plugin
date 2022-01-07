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

#include "vpux/compiler/core/barrier_scheduler.hpp"

using namespace vpux::VPURT;

bool BarrierScheduler::scheduleNumberTaskComparator::operator()(mlir::Operation* op1, mlir::Operation* op2) const {
    int64_t schedulingNumber1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(schedulingNumberAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t schedulingNumber2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(schedulingNumberAttrName).cast<mlir::IntegerAttr>().getInt());

    return schedulingNumber1 < schedulingNumber2;
}

bool BarrierScheduler::uniqueIDTaskComparator::operator()(mlir::Operation* op1, mlir::Operation* op2) const {
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

bool BarrierScheduler::barrierTransitionStructure::uniqueIDTaskComparator::operator()(mlir::Operation* op1,
                                                                                      mlir::Operation* op2) const {
    int64_t uniqueId1 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
    int64_t uniqueId2 = checked_cast<int64_t>(
            mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());

    return uniqueId1 < uniqueId2;
}
