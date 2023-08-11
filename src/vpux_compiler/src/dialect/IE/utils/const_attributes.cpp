//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

namespace vpux {
namespace IE {

mlir::ArrayAttr getIntArrayAttrValue(mlir::Value operand) {
    if (operand == nullptr) {
        return nullptr;
    }
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return nullptr;
    }
    const auto content = constOp.content();
    return getIntArrayAttr(operand.getContext(), content.getValues<int32_t>());
}

mlir::ArrayAttr getFloatArrayAttrValue(mlir::Value operand) {
    if (operand == nullptr) {
        return nullptr;
    }
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    if (constOp == nullptr) {
        return nullptr;
    }
    const auto content = constOp.content();
    return getFPArrayAttr(operand.getContext(), content.getValues<double>());
}

mlir::IntegerAttr getIntAttrValue(mlir::Value operand, mlir::PatternRewriter& rewriter) {
    if (operand == nullptr) {
        return nullptr;
    }
    auto constOp = operand.getDefiningOp<Const::DeclareOp>();
    const auto content = constOp.content();
    if (!content.isSplat()) {
        return nullptr;
    }
    const auto attrValue = content.getSplatValue<int32_t>();
    return rewriter.getI32IntegerAttr(attrValue);
}

mlir::FailureOr<Const::DeclareOp> getConstParentOp(mlir::Value input) {
    auto parent = input.getDefiningOp();
    while (parent && mlir::isa<IE::FakeQuantizeOp, IE::TransposeOp, IE::NegativeOp>(parent)) {
        parent = parent->getOperand(0).getDefiningOp();
    }
    if (parent && mlir::isa<Const::DeclareOp>(parent)) {
        return mlir::cast<Const::DeclareOp>(parent);
    }
    return mlir::failure();
}

}  // namespace IE
}  // namespace vpux
