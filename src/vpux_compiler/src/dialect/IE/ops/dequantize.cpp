//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DequantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::DequantizeOpAdaptor dequantize(operands, attrs);
    if (mlir::failed(dequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dequantize.input().getType().cast<mlir::ShapedType>();
    const auto dstElemType = dequantize.dstElemType().getValue();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}
