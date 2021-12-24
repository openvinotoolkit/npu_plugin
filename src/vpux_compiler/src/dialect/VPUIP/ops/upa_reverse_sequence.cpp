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

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(ReverseSequenceUPAOp op) {
    const auto seqShape = getShape(op.seq_length());

    if (seqShape.size() != 1) {
        return errorAt(op, "ReverseSequence second input should be 1D Tensor");
    }

    return mlir::success();
}

void vpux::VPUIP::ReverseSequenceUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                              mlir::Value seq_length, mlir::Value output, mlir::IntegerAttr seq_axis,
                                              mlir::IntegerAttr batch_axis) {
    build(builder, state, data, seq_length, output, seq_axis, batch_axis, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReverseSequenceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::ReversesequenceParamsBuilder builder(writer);
    builder.add_seq_axis(checked_cast<int32_t>(seq_axis()));
    builder.add_batch_axis(checked_cast<int32_t>(batch_axis()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReversesequenceParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseReverseSequence(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                               ArrayRef<mlir::Value> outputs,
                                                               const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 2, "UPAReverseSequence supports only 2 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAReverseSequence supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_ReversesequenceParams();
    const auto seq_axis = getIntAttr(_ctx, params->seq_axis());
    const auto batch_axis = getIntAttr(_ctx, params->batch_axis());
    return builder.create<VPUIP::ReverseSequenceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0],
                                                       seq_axis, batch_axis);
}
