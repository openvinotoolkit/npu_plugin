//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
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
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BroadcastOpAdaptor broadcast(operands, attrs);
    if (mlir::failed(broadcast.verify(loc))) {
        return mlir::failure();
    }

    auto inShape = to_small_vector(broadcast.input().getType().cast<mlir::ShapedType>().getShape());
    auto targetShape = IE::constInputToData(loc, broadcast.target_shape()).getValue();
    const auto inType = broadcast.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto broadcastMode = broadcast.mode().has_value() ? broadcast.mode().value() : IE::BroadcastType::NUMPY;

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

mlir::OpFoldResult vpux::IE::BroadcastOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    // move broadcast to const attribute.
    if (auto contentAttr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        if (!contentAttr.getBaseContent().isSplat()) {
            return nullptr;
        }
        const auto inputShape = to_small_vector(getShape(input()));
        const auto outputShape = to_small_vector(getShape(output()));
        const auto broadcastType = mode().value_or(IE::BroadcastType::NUMPY);
        SmallVector<int64_t> broadcastAxes;

        VPUX_THROW_WHEN(inputShape.size() > outputShape.size(),
                        "Invalid BroadcastOp, inputShape size {0}..outputShape size {1}", inputShape.size(),
                        outputShape.size());

        // Finds the axes over which the broadcasting rules apply. For example:
        // NUMPY and BIDIRECTIONAL: input 16x1x1, output 1x16x50x50 will return the axes [0, 2, 3]
        // EXPLICIT:                input 16, output 1x16x50x50, axesMapping 1 will return the axes [0, 2, 3]
        if (broadcastType == IE::BroadcastType::BIDIRECTIONAL || broadcastType == IE::BroadcastType::NUMPY) {
            broadcastAxes = vpux::IE::getBroadcastAxesNumpyBidirectional(inputShape, outputShape);
        } else if (broadcastType == IE::BroadcastType::EXPLICIT) {
            auto axesMapping = IE::constInputToData(getLoc(), axes_mapping()).getValue();
            broadcastAxes = vpux::IE::getBroadcastAxesExplicit(axesMapping, outputShape);
        } else {
            VPUX_THROW("Invalid broadcast type..{0}", broadcastType);
        }

        auto adjustedInputShape = inputShape;
        for (const auto& axis : broadcastAxes) {
            if (adjustedInputShape.size() < outputShape.size()) {
                adjustedInputShape.insert(adjustedInputShape.begin() + axis, 1);
            }
        }

        contentAttr = contentAttr.reshape(Shape(adjustedInputShape));
        for (const auto& dim : enumerate(outputShape)) {
            if (dim.value() > 1 && dim.value() != adjustedInputShape[dim.index()]) {
                contentAttr = contentAttr.broadcast(Dim(dim.index()), outputShape[dim.index()]);
            }
        }

        return contentAttr;
    }

    return nullptr;
}
