//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AdaptiveAvgPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AdaptiveAvgPoolOpAdaptor adaptiveAvgPool(operands, attrs);
    if (mlir::failed(adaptiveAvgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = adaptiveAvgPool.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inputType.getShape();
    auto pooledSpatialShape = IE::constInputToData(loc, adaptiveAvgPool.pooled_spatial_shape()).getValue();

    if (inputShape.size() != 3 && inputShape.size() != 4 && inputShape.size() != 5) {
        return errorAt(loc, "Input shape should have 3, 4 or 5 dimensions");
    }

    SmallVector<int64_t> outputShape;
    outputShape.push_back(inputShape[0]);
    outputShape.push_back(inputShape[1]);
    for (size_t i = 0; i < pooledSpatialShape.size(); i++) {
        outputShape.push_back(pooledSpatialShape[i]);
    }

    inferredReturnShapes.emplace_back(outputShape, inputType.getElementType());
    return mlir::success();
}
