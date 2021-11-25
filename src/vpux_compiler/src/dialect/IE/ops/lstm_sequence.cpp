//
// Copyright Intel Corporation.
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

using namespace vpux;

mlir::LogicalResult vpux::IE::LSTMSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::LSTMSequenceOpAdaptor lstm(operands, attrs);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.initialHiddenState().getType().cast<mlir::ShapedType>();
    auto outHVShape = inType.getShape().vec();
    outHVShape.insert(outHVShape.cbegin() + 2, lstm.sequenceLength().getInt());

    inferredReturnShapes.emplace_back(outHVShape, inType.getElementType());         // outputHiddenValues
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputHiddenState
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputCellState

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::IE::LSTMSequenceOp::serialize(EMU::BlobWriter& writer) {
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
