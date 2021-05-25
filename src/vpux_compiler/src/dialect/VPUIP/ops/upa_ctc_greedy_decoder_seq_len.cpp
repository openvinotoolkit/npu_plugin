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
#include "vpux/compiler/utils/subspaces.hpp"

using namespace vpux;

void vpux::VPUIP::CTCGreedyDecoderSeqLenUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                     mlir::Value input, mlir::Value sequenceLength,
                                                     mlir::Value blankIndex, mlir::Value output,
                                                     mlir::Value outputLength, mlir::UnitAttr mergeRepeated) {
    build(builder, state, input, sequenceLength, blankIndex, output, outputLength, mlir::ValueRange{},
          mlir::ValueRange{}, mergeRepeated, nullptr, nullptr);
}

mlir::LogicalResult vpux::VPUIP::CTCGreedyDecoderSeqLenUPAOp::isSupportedLayout(mlir::Operation* op,
                                                                                vpux::DataOrderInfo& info) {
    const auto ctcGreedyDecoderSeqLenOp = mlir::dyn_cast<IERT::CTCGreedyDecoderSeqLenOp>(op);
    VPUX_THROW_UNLESS(ctcGreedyDecoderSeqLenOp != nullptr, "Operation {0} is not CTCGreedyDecoderSeqLenOp",
                      op->getName());

    if (info.hasInput(0)) {
        const auto order = info.getInput(0);
        if (order == DimsOrder::CHW) {
            return mlir::success();
        }
    }

    info.setInput(0, DimsOrder::CHW);
    return mlir::failure();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CTCGreedyDecoderSeqLenUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CTCGreedyDecoderSeqLenParamsBuilder builder(writer);
    builder.add_mergeRepeated(mergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this,
                                     {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCGreedyDecoderSeqLenParams});
}
