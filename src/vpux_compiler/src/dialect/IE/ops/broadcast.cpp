//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

SmallVector<int64_t> getResultShapeBidirectional(SmallVector<int64_t>& inShape, SmallVector<int64_t>& targetShape) {
    const auto targetPaddedRank = std::max(inShape.size(), targetShape.size());

    SmallVector<int64_t> resultShape(targetPaddedRank);

    while (inShape.size() < targetPaddedRank) {
        inShape.insert(inShape.begin(), 1);
    }

    while (targetShape.size() < targetPaddedRank) {
        targetShape.insert(targetShape.begin(), 1);
    }

    for (size_t i = 0; i < targetPaddedRank; ++i) {
        VPUX_THROW_UNLESS(inShape[i] == 1 || targetShape[i] == 1 || inShape[i] == targetShape[i],
                          "Broadcast incorrect target shape. Expecting either 1 or {0}. Got {1}", inShape[i],
                          targetShape[i]);

        resultShape[i] = std::max(inShape[i], targetShape[i]);
    }

    return resultShape;
}

mlir::LogicalResult vpux::IE::BroadcastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::BroadcastOpAdaptor broadcast(operands, attrs);
    if (mlir::failed(broadcast.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = to_small_vector(broadcast.input().getType().cast<mlir::ShapedType>().getShape());
    auto targetShape = IE::constInputToData(loc, broadcast.target_shape()).getValue();
    const auto inType = broadcast.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto broadcastMode = broadcast.mode().getValue();

    SmallVector<int64_t> outShape;

    if (broadcastMode == IE::BroadcastType::NUMPY || broadcastMode == IE::BroadcastType::EXPLICIT) {
        outShape = targetShape;
    } else if (broadcastMode == IE::BroadcastType::BIDIRECTIONAL) {
        outShape = getResultShapeBidirectional(inShape, targetShape);
    }

    inferredReturnShapes.emplace_back(outShape, inType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::BroadcastOp::fold(ArrayRef<mlir::Attribute>) {
    if (input().getType() == output().getType()) {
        return input();
    }

    return nullptr;
}
