//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

void vpux::VPUIP::LSTMCellUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                        mlir::Value inputData, mlir::Value initialHiddenState,
                                        mlir::Value initialCellState, mlir::Value weights, mlir::Value biases,
                                        mlir::Value outputHiddenState, mlir::Value outputCellState) {
    build(builder, state, inputData, initialHiddenState, initialCellState, weights, biases,
            outputHiddenState, outputCellState, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, nullptr);
}

vpux::VPUIP::BlobWriter::SpecificTask vpux::VPUIP::LSTMCellUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::LSTMCellParamsBuilder builder(writer);
    builder.add_RNNForward(1);
    builder.add_nCells(1);
    const auto inputDataShape = inputData().getType().cast<mlir::ShapedType>().getShape();
    VPUX_THROW_UNLESS(inputDataShape.size() == 2, "LSTMCellUPAOp inputData shape must be 2D");
    const auto batchSize = inputDataShape[0];
    builder.add_nBatches(static_cast<int32_t>(batchSize));
    builder.add_useCellState(1);
    builder.add_outputsNumber(2);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_LSTMCellParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseLSTMCell(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                                ArrayRef<mlir::Value> outputs,
                                                                const MVCNN::UPALayerTask* /*task*/) {
    VPUX_THROW_UNLESS(inputs.size() == 5, "LSTMCell supports only 5 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 2, "LSTMCell supports only 2 output, got {0}", outputs.size());

    return builder.create<VPUIP::LSTMCellUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], outputs[1], outputs[2]);
}
