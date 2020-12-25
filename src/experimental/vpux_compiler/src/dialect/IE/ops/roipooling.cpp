//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ROIPoolingOpAdaptor roiPooling(operands, attrs);
    if (mlir::failed(roiPooling.verify(loc))) {
        return ::mlir::failure();
    }

    std::function<SmallVector<int64_t, MAX_NUM_DIMS>(mlir::ArrayAttr && arrayAttr)> convertArrayAttrToSmallVector =
            [](mlir::ArrayAttr&& arrayAttr) {
                SmallVector<int64_t, MAX_NUM_DIMS> result;
                for (auto&& a : arrayAttr)
                    result.push_back(a.dyn_cast<mlir::IntegerAttr>().getInt());
                return result;
            };

    SmallVector<int64_t, MAX_NUM_DIMS> output_size = convertArrayAttrToSmallVector(roiPooling.output_size());

    auto inTypeFeatureMap = roiPooling.input().getType().cast<mlir::RankedTensorType>();
    auto inTypeCoord = roiPooling.coords().getType().cast<mlir::RankedTensorType>();
    auto inShapeFeatureMap = inTypeFeatureMap.getShape();
    auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4)
        return mlir::LogicalResult(printTo(mlir::emitError(loc),
                                           "Dimension of the feature maps input should be 4. Got {0} D tensor",
                                           inShapeFeatureMap.size()));

    if (inShapeCoord.size() != 2)
        return mlir::LogicalResult(printTo(
                mlir::emitError(loc), "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                inShapeCoord.size()));

    if (output_size.size() != 2)
        return mlir::LogicalResult(printTo(mlir::emitError(loc),
                                           "Dimension of pooled size is expected to be equal to 2. Got {0}",
                                           output_size.size()));

    if (output_size[0] <= 0 && output_size[1] <= 0)
        return mlir::LogicalResult(printTo(mlir::emitError(loc),
                                           "Pooled size attributes pooled_h and pooled_w should should be positive."));

    SmallVector<int64_t, MAX_NUM_DIMS> output_shape;
    output_shape.push_back(inShapeCoord[0]);
    output_shape.push_back(inShapeFeatureMap[1]);
    output_shape.push_back(output_size[0]);
    output_shape.push_back(output_size[1]);

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());

    return mlir::success();
}
