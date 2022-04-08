//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AtanOp::inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::AtanOpAdaptor atan(operands, attrs);
    if (mlir::failed(atan.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = atan.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::AtanOp::serialize(EMU::BlobWriter& writer) {
    const auto atan = MVCNN::CreateAtanParams(writer);

    MVCNN::PostOpsParamsBuilder builder(writer);
    builder.add_nested_params_type(MVCNN::PostOpsNestedParams_AtanParams);
    builder.add_nested_params(atan.Union());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PostOpsParams});
}
