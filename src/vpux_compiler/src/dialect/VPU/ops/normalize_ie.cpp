//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::NormalizeIEOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               mlir::Optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NormalizeIEOpAdaptor normalize(operands, attrs);
    if (mlir::failed(normalize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = normalize.data().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::NormalizeIEOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::NormalizeParamsBuilder builder(writer);
    builder.add_eps(static_cast<float>(eps().convertToDouble()));
    builder.add_across_spatial(static_cast<int32_t>(across_spatial()));
    builder.add_channel_shared(static_cast<int32_t>(channel_shared()));

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_NormalizeParams});
}
