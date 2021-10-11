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

#include "vpux/compiler/utils/stl_extras.hpp"

#include <cassert>

using namespace vpux;

bool vpux::OpOrderCmp::operator()(mlir::Operation* lhs, mlir::Operation* rhs) const {
    assert(lhs->getBlock() == rhs->getBlock());

    return lhs->isBeforeInBlock(rhs);
}

bool vpux::ValueOrderCmp::operator()(mlir::Value lhs, mlir::Value rhs) const {
    assert(lhs.getParentBlock() == rhs.getParentBlock());

    if (lhs.isa<mlir::OpResult>() && rhs.isa<mlir::OpResult>()) {
        if (lhs.getDefiningOp() == rhs.getDefiningOp()) {
            return lhs.cast<mlir::OpResult>().getResultNumber() < rhs.cast<mlir::OpResult>().getResultNumber();
        } else {
            return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
        }
    } else if (lhs.isa<mlir::BlockArgument>() && rhs.isa<mlir::OpResult>()) {
        return true;
    } else if (lhs.isa<mlir::OpResult>() && rhs.isa<mlir::BlockArgument>()) {
        return false;
    } else {
        return lhs.cast<mlir::BlockArgument>().getArgNumber() < rhs.cast<mlir::BlockArgument>().getArgNumber();
    }
}
