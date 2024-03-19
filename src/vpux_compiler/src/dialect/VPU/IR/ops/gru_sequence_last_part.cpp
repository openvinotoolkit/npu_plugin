//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceLastPartOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceLastPartOpAdaptor gru(operands, attrs);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto initialStateType = gru.getInitialHiddenState().getType().cast<vpux::NDTypeInterface>();
    const auto outputStateType = initialStateType;
    const auto outputStateShape = outputStateType.getShape().raw();
    const auto seqLength = gru.getSeqLength();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};
    const auto middleStateType = initialStateType.changeShape(Shape(middleStateShape));

    inferredReturnShapes.push_back(middleStateType);
    inferredReturnShapes.push_back(outputStateType);

    return mlir::success();
}
