
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

using namespace vpux;

mlir::LogicalResult vpux::IE::DetectionOutputOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::DetectionOutputOpAdaptor detectionOutput(operands, attrs);
    if (mlir::failed(detectionOutput.verify(loc))) {
        return ::mlir::failure();
    }

    auto boxLogitsType = detectionOutput.box_logits().getType().cast<mlir::RankedTensorType>();

    auto num_images = boxLogitsType.getShape()[0];
    auto num_loc_classes =
            detectionOutput.attr().share_location().getValue() ? 1 : detectionOutput.attr().num_classes().getInt();
    if (num_loc_classes <= 0) {
        return mlir::LogicalResult(printTo(mlir::emitError(loc), "Number of classes should be a natural number"));
    }
    if (boxLogitsType.getShape()[1] % (num_loc_classes * 4) != 0) {
        return mlir::LogicalResult(printTo(
                mlir::emitError(loc), "For [N, C, H, W] input shape, C should be divisible by num_loc_classes * 4."));
    }
    auto num_prior_boxes = boxLogitsType.getShape()[1] / (num_loc_classes * 4);
    mlir::SmallVector<int64_t, 4> output_shape{1, 1};
    if (detectionOutput.attr().keep_top_k()[0].dyn_cast<mlir::IntegerAttr>().getInt() > 0) {
        output_shape.push_back(num_images *
                               detectionOutput.attr().keep_top_k()[0].dyn_cast<mlir::IntegerAttr>().getInt());
    } else if (detectionOutput.attr().top_k().getInt() > 0) {
        output_shape.push_back(num_images * detectionOutput.attr().top_k().getInt() *
                               detectionOutput.attr().num_classes().getInt());
    } else {
        output_shape.push_back(num_images * num_prior_boxes * detectionOutput.attr().num_classes().getInt());
    }
    output_shape.push_back(7);

    inferredReturnShapes.emplace_back(output_shape, boxLogitsType.getElementType());

    return mlir::success();
}
