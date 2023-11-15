//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FloorModOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            mlir::Optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FloorModOpAdaptor floorMod(operands, attrs);
    if (mlir::failed(floorMod.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = floorMod.input1().getType().cast<vpux::NDTypeInterface>();
    const auto in2Type = floorMod.input2().getType().cast<vpux::NDTypeInterface>();

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       floorMod.auto_broadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = in1Type.changeShape(Shape(outShapeRes.value()));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::FloorModOp::serialize(EMU::BlobWriter& writer) {
    EMU::BlobWriter::String type;
    type = writer.createString("floormod");

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}
