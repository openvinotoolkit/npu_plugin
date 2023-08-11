//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
