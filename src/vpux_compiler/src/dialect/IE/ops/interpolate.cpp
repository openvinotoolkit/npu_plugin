
// Copyright 2020 Intel Corporation.
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
