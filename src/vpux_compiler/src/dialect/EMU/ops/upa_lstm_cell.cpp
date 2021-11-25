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

#include "vpux/compiler/dialect/EMU/ops.hpp"

vpux::EMU::BlobWriter::SpecificTask vpux::EMU::LSTMCellUPAOp::serialize(EMU::BlobWriter& writer) {
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
