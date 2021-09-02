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
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(GatherUPAOp op) {

    // Axis should be 'scalar or 1D tensor', but VPUIP only supports scalar for now
    const auto axisRank = op.axis().getType().dyn_cast<mlir::ShapedType>().getRank();
    if (axisRank != 0) {
        return errorAt(op, "Only support scalar axis {0} ", axisRank);
    }

    // Axis should not exceed input rank
    const auto axisNo = op.axis().getSplatValue<int64_t>();
    const auto inShape = getShape(op.input());
    if (checked_cast<size_t>(axisNo) >= inShape.size()) {
        return errorAt(op, "Gather axis '{0}' is out of range [0,{1}]", axisNo, inShape.size()-1);
    }

    // Indices tensor should be int32
    const mlir::Type GF_INT32 = getSInt32Type(op.getContext());
    const auto idxType = op.indices().getType().cast<mlir::ShapedType>().getElementType();

    if (idxType != GF_INT32) {
        return errorAt(op, "Unsupported indices type : '{0}', only int32 is valid", idxType);
    }

    return mlir::success();
}

void vpux::VPUIP::GatherUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value indices, mlir::Value output, Const::ContentAttr axis) {
    build(builder, state, input, indices, output, axis, mlir::ValueRange{}, mlir::ValueRange{}, nullptr, false);
}

bool vpux::VPUIP::GatherUPAOp::isSupportedLayout(mlir::Operation* op, IE::DataOrderInfo&) {
    VPUX_THROW_UNLESS(mlir::isa<IE::GatherOp>(op), "Operation {0} is not Gather", op->getName());
    return true; // any layout
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::GatherUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::GatherParamsBuilder builder(writer);
    auto axisNo = axis().getSplatValue<int64_t>();
    builder.add_axis(checked_cast<uint32_t>(axisNo));
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parseGather(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                      ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size()  == 2, "GatherUPA supports only 2 inputs", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "GatherUPA supports only 1 output", outputs.size());

    const auto params = task->softLayerParams_as_GatherParams();

    const auto axisShapeType = mlir::RankedTensorType::get({}, getInt32Type(_ctx));
    SmallVector<int32_t> axisVec;
    axisVec.push_back(params->axis());
    const auto axisContent = Const::ContentAttr::get(mlir::DenseElementsAttr::get(axisShapeType, makeArrayRef(axisVec)));

    return builder.create<VPUIP::GatherUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0], axisContent);
}
