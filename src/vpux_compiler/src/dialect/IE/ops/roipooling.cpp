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

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ROIPoolingOpAdaptor roiPooling(operands, attrs);
    if (mlir::failed(roiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto output_size = parseIntArrayAttr<int64_t>(roiPooling.output_size());
    const auto inTypeFeatureMap = roiPooling.input().getType().cast<mlir::ShapedType>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = roiPooling.coords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (output_size.size() != 2) {
        return errorAt(loc, "Dimension of pooled size is expected to be equal to 2. Got {0}", output_size.size());
    }

    if (output_size[0] <= 0 && output_size[1] <= 0) {
        return errorAt(loc, "Pooled size attributes pooled_h and pooled_w should should be positive.");
    }

    SmallVector<int64_t> output_shape;
    output_shape.push_back(inShapeCoord[0]);
    output_shape.push_back(inShapeFeatureMap[1]);
    output_shape.push_back(output_size[0]);
    output_shape.push_back(output_size[1]);

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());
    return mlir::success();
}
