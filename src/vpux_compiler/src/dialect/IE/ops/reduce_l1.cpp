//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceL1Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceL1OpAdaptor reduceL1(operands, attrs);
    if (mlir::failed(reduceL1.verify(loc))) {
        return mlir::failure();
    }
    if (reduceL1.axes() != nullptr && reduceL1.axes_value().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceL1.axes() == nullptr && !reduceL1.axes_value().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceL1.input();
    const auto keepDims = reduceL1.keep_dims();

    auto axesValue = IE::extractAxes(loc, reduceL1);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceL1Op::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}

void vpux::IE::ReduceL1Op::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceL1Op>>(context);
}
