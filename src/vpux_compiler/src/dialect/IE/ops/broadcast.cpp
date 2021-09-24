
//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::FailureOr<SmallVector<int64_t>> extractIntVector(mlir::Location loc, const mlir::Value& value) {
    if (value != nullptr) {
        auto valueConst = value.getDefiningOp<Const::DeclareOp>();
        if (valueConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for interpolate attribute");
        }

        const auto valueContent = valueConst.content();
        return to_small_vector(valueContent.getValues<int64_t>());
    }
    return errorAt(loc, "Parameter were not provided");
}

SmallVector<int64_t> get_result_shape_bidirectional(ArrayRef<int64_t> inShape, SmallVector<int64_t> targetShape) {
    const auto target_padded_rank = std::max(inShape.size(), targetShape.size());

    SmallVector<int64_t> resultShape(target_padded_rank);
    SmallVector<int64_t> inShapeVec = to_small_vector(inShape);

    while (inShapeVec.size() < target_padded_rank) {
        inShapeVec.insert(inShapeVec.begin(), 1);
    }

    while (targetShape.size() < target_padded_rank) {
        targetShape.insert(targetShape.begin(), 1);
    }

    for (size_t i = 0; i < target_padded_rank; ++i) {
        VPUX_THROW_UNLESS(inShapeVec[i] == 1 || targetShape[i] == 1 || inShapeVec[i] == targetShape[i],
                          "Broadcast incorrect target shape. Expecting either 1 or {0}. Got {1}", inShapeVec[i],
                          targetShape[i]);  // this check is duplicated in core/src/op/broadcast.cpp, to remove??

        resultShape[i] = std::max(inShapeVec[i], targetShape[i]);
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

    const auto inType = broadcast.input().getType().cast<mlir::ShapedType>();
    const auto targetShape = extractIntVector(loc, broadcast.target_shape()).getValue();

    const auto broadcastMode = broadcast.mode().getValue();
    SmallVector<int64_t> outShape;

    if (broadcastMode == IE::BroadcastType::NUMPY) {
        outShape = targetShape;
    } else if (broadcastMode == IE::BroadcastType::BIDIRECTIONAL) {
        outShape = get_result_shape_bidirectional(inType.getShape(), targetShape);
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
