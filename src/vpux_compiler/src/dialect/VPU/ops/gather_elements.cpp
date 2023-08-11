//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherElementsOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherElementsOpAdaptor gatherElements(operands, attrs);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = gatherElements.indices().getType().cast<vpux::NDTypeInterface>();
    const auto inInputType = gatherElements.input().getType().cast<vpux::NDTypeInterface>();

    const auto outType = inInputType.changeShape(inIndicesType.getShape());
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::GatherElementsOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::GatherElementsParamsBuilder builder(writer);
    builder.add_axis(checked_cast<int32_t>(axis()));
    const auto paramsOff = builder.Finish();
    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_GatherElementsParams});
}
