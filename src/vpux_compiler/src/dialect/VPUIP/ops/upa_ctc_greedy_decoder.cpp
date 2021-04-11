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

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(CTCGreedyDecoderUPAOp op) {
    const auto inShape = getShape(op.input());

    if (inShape.size() != 3) {
        return errorAt(op, "Input shape should have 3 dimensions");
    }

    if (inShape[Dim(1)] != 1) {
        return errorAt(op, "Input tensor [T N C] = [{0} {1} {2}] has unsupported dimension size N != 1",
                       inShape[Dim(0)], inShape[Dim(1)], inShape[Dim(2)]);
    }

    return mlir::success();
}

void vpux::VPUIP::CTCGreedyDecoderUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                               mlir::Value sequenceLengths, mlir::Value output,
                                               mlir::UnitAttr mergeRepeated) {
    build(builder, state, input, sequenceLengths, output, mlir::ValueRange{}, mlir::ValueRange{}, mergeRepeated,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CTCGreedyDecoderUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CTCDecoderParamsBuilder builder(writer);
    builder.add_ctc_merge_repeated(mergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCDecoderParams});
}
