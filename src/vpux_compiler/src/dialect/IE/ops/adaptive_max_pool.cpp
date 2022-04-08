//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AdaptiveMaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AdaptiveMaxPoolOpAdaptor adaptiveMaxPool(operands, attrs);
    if (mlir::failed(adaptiveMaxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = adaptiveMaxPool.input().getType().cast<mlir::ShapedType>();
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
    outputShape.push_back(inputShape[0]);
    outputShape.push_back(inputShape[1]);
    for (size_t i = 0; i < pooledSpatialShape.size(); i++) {
        outputShape.push_back(pooledSpatialShape[i]);
    }

    inferredReturnShapes.emplace_back(outputShape, inputType.getElementType());
    inferredReturnShapes.emplace_back(outputShape, adaptiveMaxPool.index_element_type());

    return mlir::success();
}
