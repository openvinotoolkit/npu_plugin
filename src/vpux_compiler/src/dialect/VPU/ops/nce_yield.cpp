//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// RegionBranchTerminatorOpInterface
//

mlir::MutableOperandRange vpux::VPU::YieldOp::getMutableSuccessorOperands(mlir::Optional<unsigned> index) {
    VPUX_THROW_UNLESS(!index.hasValue(), "Can't process the index value");
    return operandsMutable();
}

//
// verifyOp
//

mlir::LogicalResult vpux::VPU::verifyOp(vpux::VPU::YieldOp op) {
    if (op->getNumOperands() < 1) {
        return errorAt(op->getLoc(), "Operation must have at least one operand");
    }

    return mlir::success();
}
