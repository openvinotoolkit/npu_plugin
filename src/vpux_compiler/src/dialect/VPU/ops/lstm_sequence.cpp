//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::LSTMSequenceOpAdaptor lstm(operands, attrs);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.initialHiddenState().getType().cast<vpux::NDTypeInterface>();
    auto outHVShape = inType.getShape().raw().vec();
    outHVShape.insert(outHVShape.cbegin() + 2, lstm.sequenceLength().getInt());

    const auto outType = inType.changeShape(Shape(outHVShape));

    inferredReturnTypes.push_back(outType);  // outputHiddenValues
    inferredReturnTypes.push_back(inType);   // outputHiddenState
    inferredReturnTypes.push_back(inType);   // outputCellState

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::LSTMSequenceOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::LSTMCellParamsBuilder builder(writer);
    builder.add_RNNForward(direction() == IE::RNNSequenceDirection::FORWARD ? 1 : 0);
    builder.add_nCells(checked_cast<int32_t>(sequenceLength()));
    const auto inputDataShape = inputData().getType().cast<mlir::ShapedType>().getShape();
    VPUX_THROW_UNLESS(inputDataShape.size() == 3, "LSTMSequenceUPAOp inputData shape must be 3D");
    const auto batchSize = inputDataShape[0];
    builder.add_nBatches(checked_cast<int32_t>(batchSize));
    builder.add_useCellState(1);
    builder.add_outputsNumber(3);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_LSTMCellParams});
}
