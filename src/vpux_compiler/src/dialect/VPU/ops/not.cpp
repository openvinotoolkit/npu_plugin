//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LogicalNotOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::LogicalNotOpAdaptor logicalNot(operands, attrs);
    if (mlir::failed(logicalNot.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = logicalNot.input1().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::LogicalNotOp::serialize(vpux::EMU::BlobWriter& writer) {
    vpux::EMU::BlobWriter::String type;
    type = writer.createString("logicalnot");

    MVCNN::EltwiseParamsBuilder builder(writer);
    builder.add_operation(type);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EltwiseParams});
}
