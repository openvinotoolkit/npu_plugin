//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ScaleShiftOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::ScaleShiftOpAdaptor scaleShift(operands, attrs);
    if (mlir::failed(scaleShift.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scaleShift.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// ScaleShift
//

void vpux::VPU::ScaleShiftOp::inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
    // Investigate performance degradation for NHWC layout
    // [Track number: E#25601]
    VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::CHW});
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::ScaleShiftOp::serialize(EMU::BlobWriter& writer) {
    const auto scaleShift = MVCNN::CreateScaleShiftParams(writer);

    MVCNN::PostOpsNestedParams opType{};
    if (weights() != nullptr && biases() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_ScaleShiftParams;
    } else if (weights() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_ScaleParams;
    } else if (biases() != nullptr) {
        opType = MVCNN::PostOpsNestedParams_BiasParams;
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(opType);
    builder.add_nested_params(scaleShift.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
