//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceFirstPartOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange, mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceFirstPartOpAdaptor gru(operands, attrs);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto inputDataType = gru.input_data().getType().cast<vpux::NDTypeInterface>();
    const auto weightsType = gru.weights().getType().cast<vpux::NDTypeInterface>();
    const auto inputDataShape = inputDataType.getShape().raw();
    const auto weightsShape = weightsType.getShape().raw();
    const auto seqLength = gru.seq_length();
    SmallVector<int64_t> outputShape = {inputDataShape[0], weightsShape[0], seqLength, weightsShape[1]};
    const auto outputType = weightsType.changeShape(Shape(outputShape));

    inferredReturnShapes.push_back(outputType);

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::GRUSequenceFirstPartOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("GRUSequenceFirstPartOp implemented just on 37xx.");
}
