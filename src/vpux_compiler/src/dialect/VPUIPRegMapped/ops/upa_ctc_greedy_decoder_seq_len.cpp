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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

// Alex: #include "vpux/compiler/dialect/VPUIPRegMapped/blob_reader.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

using namespace vpux;

void vpux::VPUIPRegMapped::CTCGreedyDecoderSeqLenUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                                              mlir::Value input, mlir::Value sequenceLength,
                                                              mlir::Value blankIndex, mlir::Value output,
                                                              mlir::Value outputLength, mlir::UnitAttr mergeRepeated) {
    build(builder, state, input, sequenceLength, blankIndex, output, outputLength, mlir::ValueRange{},
          mlir::ValueRange{}, mergeRepeated, nullptr, nullptr);
}

void vpux::VPUIPRegMapped::CTCGreedyDecoderSeqLenUPAOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    IE::fillDefaultLayoutInfo(info);
}

// VPUIPRegMapped::BlobWriter::SpecificTask
// vpux::VPUIPRegMapped::CTCGreedyDecoderSeqLenUPAOp::serialize(VPUIPRegMapped::BlobWriter& writer) {
void vpux::VPUIPRegMapped::CTCGreedyDecoderSeqLenUPAOp::serialize(std::vector<char>& buffer) {
    /*
    MVCNN::CTCGreedyDecoderSeqLenParamsBuilder builder(writer);
    builder.add_mergeRepeated(mergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this,
                                 {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCGreedyDecoderSeqLenParams});
    */

    (void)buffer;
}

/*
mlir::Operation* vpux::VPUIPRegMapped::BlobReader::parseCTCGreedyDecoderSeqLen(mlir::OpBuilder& builder,
                                                                               ArrayRef<mlir::Value> inputs,
                                                                               ArrayRef<mlir::Value> outputs,
                                                                               const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 3, "UPACTCGreedyDecoderSeqLen supports only 3 inputs, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 2, "UPACTCGreedyDecoderSeqLen supports only 2 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_CTCGreedyDecoderSeqLenParams();
    return builder.create<VPUIPRegMapped::CTCGreedyDecoderSeqLenUPAOp>(
            mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], inputs[2], outputs[0], outputs[1],
            params->mergeRepeated() ? mlir::UnitAttr::get(_ctx) : nullptr);
}
*/
