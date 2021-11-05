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

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::EMU::verifyOp(CTCGreedyDecoderUPAOp op) {
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

EMU::BlobWriter::SpecificTask vpux::EMU::CTCGreedyDecoderUPAOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::CTCDecoderParamsBuilder builder(writer);
    builder.add_ctc_merge_repeated(mergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCDecoderParams});
}
