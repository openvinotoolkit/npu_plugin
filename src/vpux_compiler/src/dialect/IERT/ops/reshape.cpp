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

#include "vpux/compiler/dialect/IERT/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

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

mlir::OpFoldResult vpux::IERT::GenericReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    if (input().getDefiningOp<GenericReshapeOp>() != nullptr) {
        return output();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reshape(getShape(output()));
    }

    return nullptr;
}
