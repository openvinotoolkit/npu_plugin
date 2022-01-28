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
