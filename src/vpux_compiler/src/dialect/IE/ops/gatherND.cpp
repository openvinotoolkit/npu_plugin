//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GatherNDOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatherNDOpAdaptor gatherND(operands, attrs);
    if (mlir::failed(gatherND.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gatherND.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();
    const auto indicesShape = gatherND.indices().getType().cast<mlir::ShapedType>().getShape();

    const auto batchDims = gatherND.batch_dims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());

    SmallVector<int64_t> outShape;
    outShape.append(indicesShape.begin(), indicesShape.end() - 1);
    if (batchDims + lastIndices != inputRank) {
        outShape.append(inputShape.begin() + batchDims + lastIndices, inputShape.end());
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::GatherNDOp::verify() {
    const auto op = getOperation();
    const auto inType = input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();
    const auto indicesShape = indices().getType().cast<mlir::ShapedType>().getShape();

    const auto batchDims = batch_dims();
    const auto lastIndices = indicesShape.back();
    const auto inputRank = static_cast<int64_t>(inputShape.size());
    const auto indicesRank = static_cast<int64_t>(indicesShape.size());

    if (batchDims >= inputRank) {
        return errorAt(op, "batch_dims {0} exceeds input rank {1}", batchDims, inputRank);
    }

    if (batchDims >= indicesRank) {
        return errorAt(op, "batch_dims {0} exceeds indices rank {1}", batchDims, inputRank);
    }

    if (batchDims + lastIndices > inputRank) {
        return errorAt(op, "Slice index is out of bound");
    }

    for (size_t i = 0; i < static_cast<size_t>(batchDims); i++) {
        if (inputShape[i] != indicesShape[i]) {
            return errorAt(op, "Batch dimensions of data and indices must be the same");
        }
    }

    return mlir::success();
}
