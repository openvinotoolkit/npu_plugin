//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceLastPartOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange, mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceLastPartOpAdaptor gru(operands, attrs);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto initialStateType = gru.initial_hidden_state().getType().cast<vpux::NDTypeInterface>();
    const auto outputStateType = initialStateType;
    const auto outputStateShape = outputStateType.getShape().raw();
    const auto seqLength = gru.seq_length();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};
    const auto middleStateType = initialStateType.changeShape(Shape(middleStateShape));

    inferredReturnShapes.push_back(middleStateType);
    inferredReturnShapes.push_back(outputStateType);

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::GRUSequenceLastPartOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("GRUSequenceLastPartOp implemented just on 37xx.");
}
