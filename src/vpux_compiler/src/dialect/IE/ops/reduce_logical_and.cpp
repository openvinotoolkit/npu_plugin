//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceLogicalAndOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceLogicalAndOpAdaptor reduceLogicalAnd(operands, attrs);
    if (mlir::failed(reduceLogicalAnd.verify(loc))) {
        return mlir::failure();
    }
    if (reduceLogicalAnd.axes() != nullptr && reduceLogicalAnd.axes_value().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceLogicalAnd.axes() == nullptr && !reduceLogicalAnd.axes_value().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceLogicalAnd.input();
    const auto keepDims = reduceLogicalAnd.keep_dims();

    auto axesValue = IE::extractAxes(loc, reduceLogicalAnd);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceLogicalAndOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

void vpux::IE::ReduceLogicalAndOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceLogicalAndOp>>(context);
}
