//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IERT::verifyOp(GenericReshapeOp op) {
    const auto inType = op.input().getType().cast<mlir::MemRefType>();
    const auto outType = op.output().getType().cast<mlir::MemRefType>();

    if (inType.getNumElements() != outType.getNumElements()) {
        return errorAt(op, "Reshape input and output must have the same number of elements");
    }

    const auto inDimsOrder = DimsOrder::fromType(inType);
    const auto outDimsOrder = DimsOrder::fromType(outType);

    if (!inDimsOrder.isIdentity()) {
        return errorAt(op, "Only identity DimsOrder is supported for input");
    }
    if (!outDimsOrder.isIdentity()) {
        return errorAt(op, "Only identity DimsOrder is supported for output");
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

mlir::Value vpux::IERT::GenericReshapeOp::getViewSource() {
    return input();
}

mlir::OpFoldResult vpux::IERT::GenericReshapeOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    if (input().getDefiningOp<GenericReshapeOp>() != nullptr) {
        return output();
    }

    return nullptr;
}
