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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AdaptivePoolOpAdaptor::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::AdaptivePoolOpAdaptor adaptivePool(operands, attrs);
    if (mlir::failed(adaptivePool.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeFeatureMap = adaptivePool.input1().getType().cast<mlir::ShapedType>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();
    const auto inTypePooled = adaptivePool.input2().getType().cast<mlir::ShapedType>();
    const auto inShapePooled = inTypePooled.getShape();


    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapePooled.size() != 1) {
        return errorAt(loc, "Dimension of the pooled shape input with box coordinates should be 1. Got {0} D tensor",
                       inShapePooled.size());
    }


    SmallVector<int64_t> output_shape;
    output_shape.push_back(inShapeFeatureMap[0]);
    output_shape.push_back(inShapeFeatureMap[1]);

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());
    return mlir::success();
}

