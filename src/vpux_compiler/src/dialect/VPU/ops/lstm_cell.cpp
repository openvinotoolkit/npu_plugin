//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMCellOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::LSTMCellOpAdaptor lstm(operands, attrs);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.initialHiddenState().getType().cast<vpux::NDTypeInterface>();

    inferredReturnTypes.push_back(inType);  // outputHiddenState
    inferredReturnTypes.push_back(inType);  // outputCellState

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::LSTMCellOp::serialize(EMU::BlobWriter& writer) {
    const auto inputDataShape = inputData().getType().cast<mlir::ShapedType>().getShape();
    VPUX_THROW_UNLESS(inputDataShape.size() == 2, "LSTMCellUPAOp inputData shape must be 2D");

    const auto batchSize = inputDataShape[0];

    MVCNN::LSTMCellParamsBuilder builder(writer);
    builder.add_RNNForward(1);
    builder.add_nCells(1);
    builder.add_nBatches(static_cast<int32_t>(batchSize));
    builder.add_useCellState(1);
    builder.add_outputsNumber(2);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_LSTMCellParams});
}
