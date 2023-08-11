//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LayoutCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LayoutCastOpAdaptor overrideLayout(operands, attrs);
    if (mlir::failed(overrideLayout.verify(loc))) {
        return mlir::failure();
    }

    const auto outAffineMap = overrideLayout.dst_order();
    const auto inType = overrideLayout.input().getType().cast<vpux::NDTypeInterface>();
    const auto outType = inType.changeDimsOrder(DimsOrder::fromAffineMap(outAffineMap));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

EMU::BlobWriter::SpecificTask vpux::VPU::LayoutCastOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::CopyParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CopyParams});
}

//
// verify
//

mlir::LogicalResult vpux::VPU::LayoutCastOp::verify() {
    const auto outAffineMap = dst_order();
    const auto inType = input().getType().cast<vpux::NDTypeInterface>();
    if (inType.getRank() != outAffineMap.getNumDims()) {
        return errorAt(*this, "Cannot apply {0} map to {1}.", outAffineMap, inType.getShape());
    }

    return mlir::success();
}
