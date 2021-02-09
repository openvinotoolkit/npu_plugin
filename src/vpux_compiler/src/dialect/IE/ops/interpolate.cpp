
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

mlir::LogicalResult vpux::IE::InterpolateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::InterpolateOpAdaptor interpolate(operands, attrs);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = interpolate.input().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    auto targetShapeConst = interpolate.target_shape().getDefiningOp<ConstantInterface>();
    if (targetShapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for target_shape");
    }

    const auto targetShape = targetShapeConst.getContent().getValues<int64_t>();

    if (targetShape.size() != interpolate.attr().axes().size()) {
        return errorAt(loc,
                       "Num of elements in traget shape tensor: {0} should be equal to number of indices in axes: {1}",
                       targetShape.size(), interpolate.attr().axes().size());
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.emplace_back(inputShape[i]);
    }

    auto targetShapeIter = targetShape.begin();
    for (const auto& i : interpolate.attr().axes()) {
        outShape[i.cast<mlir::IntegerAttr>().getInt()] = *targetShapeIter++;
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    return mlir::success();
}
