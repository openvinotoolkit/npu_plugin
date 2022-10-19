//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult VPUIP::verifyOp(VPUIP::GenericReshapeOp op) {
    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.output().getType().cast<vpux::NDTypeInterface>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "Reshape input and output must have the same number of elements");
    }

    const auto inReqs = StrideReqs::compact(inType.getRank());
    const auto outReqs = StrideReqs::compact(outType.getRank());

    if (!inReqs.checkStrides(inType)) {
        return errorAt(op, "Input strides do not match requirements '{0}'", inType);
    }
    if (!outReqs.checkStrides(outType)) {
        return errorAt(op, "Output strides do not match requirements '{0}'", inType);
    }

    return mlir::success();
}

mlir::Value VPUIP::GenericReshapeOp::getViewSource() {
    return input();
}

mlir::OpFoldResult VPUIP::GenericReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    if (input().getDefiningOp<VPUIP::GenericReshapeOp>() != nullptr) {
        return output();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reshape(getShape(output()));
    }

    return nullptr;
}
