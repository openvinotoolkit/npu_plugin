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

#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

vpux::EMU::BlobWriter::SpecificTask vpux::EMU::LSTMSequenceUPAOp::serialize(EMU::BlobWriter& writer) {
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
