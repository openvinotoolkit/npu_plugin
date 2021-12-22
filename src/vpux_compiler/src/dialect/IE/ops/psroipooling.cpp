
//     const auto inType = gelu.input().getType().cast<mlir::ShapedType>();
//     inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

//     return mlir::success();
// }


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

mlir::LogicalResult vpux::IE::PSROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PSROIPoolingOpAdaptor psroiPooling(operands, attrs);
    if (mlir::failed(psroiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto output_dim = psroiPooling.output_dim().getInt();
    const auto spatial_scale = psroiPooling.spatial_scale().getFP();
    const auto group_size = psroiPooling.group_size().getInt();
    const auto spatial_bins_x = psroiPooling.spatial_bins_x().getInt();
    const auto spatial_bins_y = psroiPooling.spatial_bins_y().getInt();
    const auto inTypeFeatureMap = psroiPooling.input().getType().cast<mlir::ShapedType>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = psroiPooling.coords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (output_dim <= 0) {
        return errorAt(loc, "Pooled size attribute output_dim should be positive.");
    }

    if (group_size <= 0) {
        return errorAt(loc, "Group size attribute group_size should be positive.");
    }

    if (spatial_scale <= 0) {
        return errorAt(loc, "Spatial scale factor attribute spatial_scale should be positive.");
    }

    if (spatial_bins_x <= 0) {
        return errorAt(loc, "Number of bins to divide attribute spatial_bins_x should be positive.");
    }

    if (spatial_bins_y <= 0) {
        return errorAt(loc, "Number of bins to divide attribute spatial_bins_y should be positive.");
    }


    SmallVector<int64_t> output_shape;
    output_shape.push_back(inShapeCoord[0]);
    output_shape.push_back(output_dim);
    output_shape.push_back(group_size); //pooled_w
    output_shape.push_back(group_size); //pooled_h

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());
    return mlir::success();
}