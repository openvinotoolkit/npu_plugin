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

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(CumSumUPAOp op) {
    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    const auto inType = op.input().getType().cast<mlir::ShapedType>().getElementType();
    const auto outType = op.output().getType().cast<mlir::ShapedType>().getElementType();
    const auto axisNo = op.axis().getValue();

    if (checked_cast<size_t>(abs(axisNo)) >= inShape.size()) {
        return errorAt(op, "Axis '{0}' is out of range [-{1},{1}]", axisNo, inShape.size() - 1);
    }

    if (inShape.size() != outShape.size()) {
        return errorAt(op, "Input shape should be same as output shape, got 'inShape: {0}' vs. 'outShape: {1}'");
    }

    if (inType != outType) {
        return errorAt(op, "Input type should be same as output type, got 'inType: {0}' vs. 'outType: {1}'");
    }

    return mlir::success();
}

void vpux::VPUIP::CumSumUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value output, mlir::IntegerAttr axis, mlir::BoolAttr exclusive,
                                     mlir::BoolAttr reverse) {
    build(builder, state, input, output, axis, exclusive, reverse, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CumSumUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CumSumParamsBuilder builder(writer);
    builder.add_axis(axis().getValueOr(0));
    builder.add_exclusive(exclusive().getValueOr(false));
    builder.add_reverse(reverse().getValueOr(false));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CumSumParams});
}
