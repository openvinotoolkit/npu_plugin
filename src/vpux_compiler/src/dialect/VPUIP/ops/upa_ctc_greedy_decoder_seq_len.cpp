//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
