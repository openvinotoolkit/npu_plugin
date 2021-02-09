//
// Copyright 2020 Intel Corporation.
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
