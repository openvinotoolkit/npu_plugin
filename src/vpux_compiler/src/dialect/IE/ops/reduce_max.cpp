//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReduceMaxOpAdaptor reduceMax(operands, attrs);
    if (mlir::failed(reduceMax.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceMax.input();
    const auto keepDims = reduceMax.keep_dims() != nullptr;
    auto axes = IE::constInputToData(loc, reduceMax.axes()).getValue();

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axes, inferredReturnShapes);
}
