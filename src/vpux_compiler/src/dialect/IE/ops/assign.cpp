//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AssignOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AssignOpAdaptor assign(operands, attrs);
    if (mlir::failed(assign.verify(loc))) {
        return mlir::failure();
    }

    const auto rankedInType = assign.input().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = IE::getTensorAttr(rankedInType);
    inferredReturnShapes.emplace_back(rankedInType.getShape(), rankedInType.getElementType(), outDesc);

    return mlir::success();
}
