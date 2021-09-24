//
// Created by liuhao on 2021/9/24.
//
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/blob_reader.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::ReduceUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                     mlir::Value axes, mlir::Value output, mlir::BoolAttr keep_dims,
                                     VPUIP::ReduceLayerTypeAttr type) {
    auto axesOp = axes.getDefiningOp<Const::DeclareOp>();
    auto axesType = axes.getType().dyn_cast<mlir::MemRefType>();
    if (axesType == nullptr) {
        VPUX_THROW("Axes type is not MemRefType");
    }

    if (axesType.getElementType().isSignedInteger(64)) {
        auto newElementType = builder.getIntegerType(32, true);
        auto newAxesType = changeElemType(axesType, newElementType);
        auto newAxesContent = axesOp.contentAttr().convertElemType(newElementType);
        axes = builder.create<Const::DeclareOp>(state.location, newAxesType, newAxesContent);
    }
    build(builder, state, input, axes, output, mlir::ValueRange{}, mlir::ValueRange{}, keep_dims, type, nullptr,
          nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::ReduceUPAOp::serialize(VPUIP::BlobWriter& writer) {
    VPUIP::BlobWriter::String type;
    switch (this->type()) {
    case VPUIP::ReduceLayerType::MIN:
        type = writer.createString("min");
        break;
    default:
        VPUX_THROW("Unsupported ReduceLayerType {0}", this->type());
    }

    const auto keepDims = keep_dimsAttr().getValue();

    MVCNN::ReduceParamsBuilder builder(writer);
    builder.add_operation(type);
    builder.add_keep_dims(keepDims);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

