// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::PSROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PSROIPoolingOpAdaptor psroiPooling(operands, attrs);
    if (mlir::failed(psroiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputDim = psroiPooling.output_dim();
    const auto groupSize = psroiPooling.group_size();
    const auto inTypeFeatureMap = psroiPooling.input().getType().cast<mlir::ShapedType>();
    const auto inTypeCoord = psroiPooling.coords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (outputDim <= 0) {
        return errorAt(loc, "Pooled size attribute outputDim should be positive.");
    }

    if (groupSize <= 0) {
        return errorAt(loc, "Group size attribute groupSize should be positive.");
    }

    SmallVector<int64_t> outputShape({inShapeCoord[0],  // num_rois
                                      outputDim,        // output channel number
                                      groupSize,        // pooled_w
                                      groupSize});      // pooled_h

    inferredReturnShapes.emplace_back(outputShape, inTypeFeatureMap.getElementType());
    return mlir::success();
}
