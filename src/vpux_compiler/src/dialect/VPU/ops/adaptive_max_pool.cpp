//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AdaptiveMaxPoolOp::inferReturnTypes(
        mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    VPU::AdaptiveMaxPoolOpAdaptor adaptiveMaxPool(operands, attrs);
    if (mlir::failed(adaptiveMaxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = adaptiveMaxPool.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    if (inputShape.size() != 3 && inputShape.size() != 4 && inputShape.size() != 5) {
        return errorAt(loc, "Input shape should be 3D, 4D or 5D. Got {0}D", inputShape.size());
    }

    auto spatialDimData = IE::constInputToData(loc, adaptiveMaxPool.pooled_spatial_shape());
    if (mlir::failed(spatialDimData)) {
        return mlir::failure();
    }

    auto pooledSpatialShape = spatialDimData.getValue();

    if (inputShape.size() != 2 + pooledSpatialShape.size()) {
        return errorAt(loc, "Input shape is not compatible with pooled shape size. Got {0}D and size {1}",
                       inputShape.size(), pooledSpatialShape.size());
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inputShape.raw()[0]);
    outputShape.push_back(inputShape.raw()[1]);
    for (size_t i = 0; i < pooledSpatialShape.size(); i++) {
        outputShape.push_back(pooledSpatialShape[i]);
    }

    const auto outType = inputType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    const auto outType1 = outType.changeElemType(adaptiveMaxPool.index_element_type());
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}

//
// serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::AdaptiveMaxPoolOp::serialize(EMU::BlobWriter& writer) {
    MVCNN::AdaptivePoolParamsBuilder builder(writer);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_AdaptivePoolParams});
}
