//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

namespace IE {

mlir::LogicalResult inferReduceReturnTypeComponents(mlir::Location loc, mlir::Value input, bool keepDims,
                                                    SmallVector<int64_t>& axes,
                                                    SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes);
DimsOrder calculateReducedOutputLayout(const DimsOrder& inputDimOrder, const SmallVector<int64_t>& axes);

template <typename ReduceOp>
SmallVector<int64_t> extractAxes(mlir::Location loc, ReduceOp reduceOp) {
    SmallVector<int64_t> axesValue;
    if (reduceOp.axes() != nullptr) {
        auto axes = constInputToData(loc, reduceOp.axes());
        axesValue = axes.value();
    } else if (reduceOp.axes_value().has_value()) {
        axesValue = parseIntArrayAttr<int64_t>(reduceOp.axes_value().value());
    }
    return axesValue;
}

namespace {
//
// ConvertConstToAttr
//
template <typename ReduceOp>
class ConvertConstToAttr final : public mlir::OpRewritePattern<ReduceOp> {
public:
    ConvertConstToAttr(mlir::MLIRContext* ctx): mlir::OpRewritePattern<ReduceOp>(ctx) {
        this->setDebugName("ReduceOp::ConvertConstToAttr");
    }

private:
    mlir::LogicalResult matchAndRewrite(ReduceOp reduceOp, mlir::PatternRewriter& rewriter) const final;
};

template <typename ReduceOp>
mlir::LogicalResult ConvertConstToAttr<ReduceOp>::matchAndRewrite(ReduceOp reduceOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    // check if input was already converted to Attr
    if (reduceOp.axes_value().has_value()) {
        return mlir::failure();
    }

    // convert axes into attribute
    const auto axesContent = reduceOp.axes().template getDefiningOp<Const::DeclareOp>().getContent();
    auto axesValue = to_small_vector(axesContent.template getValues<int64_t>());
    const auto axesAttr = getIntArrayAttr(reduceOp.getContext(), makeArrayRef(axesValue));

    // rewrite layer pattern
    rewriter.replaceOpWithNewOp<ReduceOp>(reduceOp, reduceOp.input(), nullptr, axesAttr, reduceOp.keep_dims());

    return mlir::success();
}

}  // namespace
}  // namespace IE
}  // namespace vpux
