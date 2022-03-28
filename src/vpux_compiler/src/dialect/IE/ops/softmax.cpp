//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SoftMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SoftMaxOpAdaptor softMax(operands, attrs);
    if (mlir::failed(softMax.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = softMax.input().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::SoftMaxOp::fold(ArrayRef<mlir::Attribute>) {
    const auto inType = input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    const auto axis = checked_cast<size_t>(axisInd());
    VPUX_THROW_UNLESS(axis < inShape.size(), "Wrong axis idx {0} for {1} dim tensor", axis, inShape.size());

    if (inShape[axis] > 1) {
        return nullptr;
    }

    const auto valueType = mlir::RankedTensorType::get(inShape, mlir::Float32Type::get(getContext()));
    const auto baseContent = Const::ContentAttr::get(mlir::DenseElementsAttr::get(valueType, 1.0f));

    return baseContent.convertElemType(output().getType().cast<mlir::ShapedType>().getElementType());
}
