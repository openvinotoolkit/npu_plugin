//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"

using namespace vpux;

mlir::Value VPUIP::PermuteCastOp::getViewSource() {
    return source();
}

//
// fold
//

mlir::OpFoldResult vpux::VPUIP::PermuteCastOp::fold(ArrayRef<mlir::Attribute> operands) {
    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reorder(DimsOrder::fromAffineMap(dst_order()));
    }

    return nullptr;
}

mlir::LogicalResult VPUIP::verifyOp(VPUIP::PermuteCastOp op) {
    auto distributedInType = op.source().getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto distributedOutType = op.result().getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (distributedInType && distributedOutType) {
        if (!isCompatibleForDistributedInputOutput(op, distributedInType, distributedOutType)) {
            return errorAt(op, "PermuteCast input and output must have the same distribution mode");
        }
    }
    const auto inType = op.source().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.result().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "PermuteCast input and output must have the same number of elements");
    }

    return mlir::success();
}
