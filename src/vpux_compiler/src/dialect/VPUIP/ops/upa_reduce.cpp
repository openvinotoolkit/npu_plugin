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
    case VPUIP::ReduceLayerType::LOGICALAND:
        type = writer.createString("logicaland");
        break;
    default:
        VPUX_THROW("Unsupported ReduceLayerType {0}", this->type());
    }

    MVCNN::ReduceParamsBuilder builder(writer);
    builder.add_operation(type);

    const bool keepDims = (keep_dimsAttr() == nullptr) ? false : true;
    builder.add_keep_dims(keepDims);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_ReduceParams});
}

// mlir::Operation* vpux::VPUIP::BlobReader::parseReduce(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
//                                                       ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
//     VPUX_THROW_UNLESS(inputs.size() == 2, "UPAReduce supports only 2 input, got {0}", inputs.size());
//     VPUX_THROW_UNLESS(outputs.size() == 1, "UPAReduce supports only 1 output, got {0}", outputs.size());
//     const auto params = task->softLayerParams_as_ReduceParams();
//     const auto typeStr = params->operation()->str();
//     VPUIP::ReduceLayerType type;
//     if (typeStr == std::string("logicaland")) {
//         type = VPUIP::ReduceLayerType::LOGICALAND;
//     } else {
//         VPUX_THROW("Unsupported ReduceLayerType {0}", typeStr);
//     }

//     const auto keep_dims = params->keep_dims() ? mlir::UnitAttr::get(_ctx) : nullptr;
//     return builder.create<VPUIP::ReduceUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], inputs[1], outputs[0], keep_dims,
//                                               VPUIP::ReduceLayerTypeAttr::get(_ctx, type));
// }
