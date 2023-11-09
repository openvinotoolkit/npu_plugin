//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"

namespace vpux {
namespace IE {

SmallVector<int64_t> getBroadcastAxesNumpyBidirectional(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape) {
    SmallVector<int64_t> broadcastAxes;
    const auto startAxis = static_cast<int64_t>(outputShape.size()) - static_cast<int64_t>(inputShape.size());
    VPUX_THROW_UNLESS(startAxis >= 0, "Broadcast axes not known deterministically");
    for (int64_t i = 0; i < static_cast<int64_t>(outputShape.size()); i++) {
        if (i < startAxis || outputShape[i] != inputShape[i - startAxis]) {
            broadcastAxes.push_back(i);
        }
    }
    return broadcastAxes;
}

SmallVector<int64_t> getBroadcastAxesExplicit(ArrayRef<int64_t> axesMapping, ArrayRef<int64_t> outputShape) {
    SmallVector<int64_t> broadcastAxes(outputShape.size());
    std::iota(broadcastAxes.begin(), broadcastAxes.end(), 0);
    for (auto i = axesMapping.rbegin(); i != axesMapping.rend(); ++i) {
        broadcastAxes.erase(broadcastAxes.begin() + *i);
    }
    return broadcastAxes;
}

Const::DeclareOp createShapeConstForBroadCast(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx,
                                              mlir::Location loc, ShapeRef shape) {
    auto intType = getSInt64Type(ctx);
    const auto shapeStorageType = mlir::RankedTensorType::get({static_cast<int64_t>(shape.size())}, intType);
    const auto shapeDenseAttr = mlir::DenseElementsAttr::get(shapeStorageType, shape.raw());
    auto newContentAttr = Const::ContentAttr::get(shapeDenseAttr).convertElemType(getSInt32Type(ctx));
    return rewriter.create<Const::DeclareOp>(loc, shapeStorageType, newContentAttr);
}

}  // namespace IE
}  // namespace vpux
