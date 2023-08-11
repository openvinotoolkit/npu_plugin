//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceSumOpAdaptor reduceSum(operands, attrs);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceSum.input();
    const auto keepDims = reduceSum.keep_dims();
    auto axes = IE::constInputToData(loc, reduceSum.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    auto axesValue = axes.getValue();

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

mlir::LogicalResult vpux::IE::ReduceSumOp::verify() {
    const auto op = getOperation();
    const auto opAxes = axes().getType().dyn_cast<mlir::RankedTensorType>();

    if (opAxes == nullptr) {
        return errorAt(op, "Axes is not a 'RankedTensorType', got '{0}'", opAxes);
    }

    const auto axesRank = opAxes.getRank();

    // The axes input must be a scalar or 1D tensor
    if (axesRank > 1) {
        return errorAt(op,
                       "Operation has unsupported tensor rank '{0}' for axes, it must be either a scalar or 1D tensor",
                       axesRank);
    }
    // The axes input must have integer type.
    if (!opAxes.getElementType().isa<mlir::IntegerType>()) {
        return errorAt(op, " Axes input must have integer element type but actual element type is '{0}'",
                       opAxes.getElementType());
    }

    // The axes input must contain unique elements
    auto axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(axes()));
    llvm::sort(axesVec);
    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(op, "Axes values should be unique");
    }

    return mlir::success();
}
