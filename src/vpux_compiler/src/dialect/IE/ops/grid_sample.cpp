//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GridSampleOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GridSampleOpAdaptor gridSample(operands, attrs);

    if (mlir::failed(gridSample.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gridSample.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    const auto gridType = gridSample.grid().getType().cast<mlir::ShapedType>();
    const auto gridShape = gridType.getShape();

    SmallVector<int64_t> outShape = {inShape[0], inShape[1], gridShape[1], gridShape[2]};

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
