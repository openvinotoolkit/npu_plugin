//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange, mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceOpAdaptor gru(operands, attrs);
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

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::GRUSequenceOp::serialize(EMU::BlobWriter& writer) {
    const auto inputDataShape = input_data().getType().cast<mlir::ShapedType>().getShape();
    VPUX_THROW_UNLESS(inputDataShape.size() == 3, "GRUSequenceUPAOp inputData shape must be 3D");

    const auto batchSize = inputDataShape[0];

    MVCNN::GRUCellParamsBuilder builder(writer);
    builder.add_hiddenSize(static_cast<int32_t>(hidden_size()));
    builder.add_batchSize(static_cast<int32_t>(batchSize));
    builder.add_sequenceLength(static_cast<int32_t>(seq_length()));
    builder.add_clip(checked_cast<float>(clipAttr().getValueAsDouble()));
    builder.add_direction(direction() == IE::RNNSequenceDirection::FORWARD ? 0 : 1);
    builder.add_linearBeforeReset(checked_cast<bool>(should_linear_before_reset()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GRUCellParams});
}
