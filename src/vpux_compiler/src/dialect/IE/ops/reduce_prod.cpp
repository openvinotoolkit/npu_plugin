//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceProdOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceProdOpAdaptor reduceProd(operands, attrs);
    if (mlir::failed(reduceProd.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceProd.input();
    const auto keepDims = reduceProd.keep_dims();
    auto axes = IE::constInputToData(loc, reduceProd.axes());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axes.getValue(), inferredReturnShapes);
}
