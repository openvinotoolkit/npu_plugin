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

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(SoftMaxUPAOp op) {
    const auto inShape = getShape(op.input());
    const auto axis = op.getAxisDim();

    if (inShape[axis] == 1) {
        return errorAt(op, "Softmax on 1 element doesn't make sense (dim along the 'axis' equal 1)");
    }

    const auto cmxSizeLimit = Byte(SHAVE_LIB_DATA_SIZE) - Byte(8 * FP16_SIZE);
    if (Byte(inShape[axis] * FP16_SIZE) > cmxSizeLimit) {
        return errorAt(op, "Axis '{0}' dimension '{1}' exceeds local CMX buffer limitation '{2}'", axis, inShape[axis],
                       cmxSizeLimit);
    }

    return mlir::success();
}

void vpux::VPUIP::SoftMaxUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                      mlir::Value output, mlir::IntegerAttr axisInd) {
    build(builder, state, input, output, mlir::ValueRange{}, mlir::ValueRange{}, axisInd, nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::SoftMaxUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto axisDim = getAxisDim();

    MVCNN::SoftmaxParamsBuilder builder(writer);
    builder.add_axis(checked_cast<uint32_t>(axisDim.ind()));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_SoftmaxParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseSoftmax(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPASoftMax supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPASoftMax supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_SoftmaxParams();
    const auto axis = getIntAttr(_ctx, params->axis());
    return builder.create<VPUIP::SoftMaxUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0], axis);
}
