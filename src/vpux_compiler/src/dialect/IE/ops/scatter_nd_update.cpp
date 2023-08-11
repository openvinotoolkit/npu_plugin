//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ScatterNDUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScatterNDUpdateOpAdaptor scatter(operands, attrs);
    if (mlir::failed(scatter.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatter.input().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = IE::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

void vpux::IE::ScatterNDUpdateOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto inType = input().getType().cast<mlir::RankedTensorType>();
    info.setInput(0, DimsOrder::fromNumDims(inType.getShape().size()));
}
