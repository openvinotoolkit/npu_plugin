//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::EmbeddingBagPackedSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::EmbeddingBagPackedSumOpAdaptor embeddingBag(operands, attrs);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto embTableType = embeddingBag.emb_table().getType().cast<mlir::ShapedType>();
    const auto embTableShape = embTableType.getShape();
    const auto indicesShape = embeddingBag.indices().getType().cast<mlir::ShapedType>().getShape();
    int64_t batchSize = indicesShape[0];

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < embTableShape.size(); i++) {
        outShape.emplace_back(embTableShape[i]);
    }
    outShape[0] = batchSize;

    inferredReturnShapes.emplace_back(outShape, embTableType.getElementType());
    return mlir::success();
}
