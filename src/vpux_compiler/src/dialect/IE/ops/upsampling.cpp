//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::UpsamplingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::UpsamplingOpAdaptor upsampling(operands, attrs);
    if (mlir::failed(upsampling.verify(loc))) {
        return mlir::failure();
    }

    auto padLVector = parseIntArrayAttr<int32_t>(upsampling.pad_l());
    auto padRVector = parseIntArrayAttr<int32_t>(upsampling.pad_r());
    auto upsamplingFactorVector = parseIntArrayAttr<int32_t>(upsampling.upsampling_factor());

    const auto inType = upsampling.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    VPUX_THROW_UNLESS(inShape.size() == 4, "Upsampling supports only 4D input tensor");
    VPUX_THROW_UNLESS(padLVector.size() == 3, "Upsampling supports pads only for 3 axes");
    VPUX_THROW_UNLESS(padRVector.size() == 3, "Upsampling supports pads only for 3 axes");
    VPUX_THROW_UNLESS(upsamplingFactorVector.size() == 3, "Upsampling supports factors only for 3 axes");

    SmallVector<int64_t> outputShape{
            inShape[0],
            inShape[1] + (inShape[1] - 1) * (upsamplingFactorVector[2] - 1) + padLVector[2] + padRVector[2],
            inShape[2] + (inShape[2] - 1) * (upsamplingFactorVector[1] - 1) + padLVector[1] + padRVector[1],
            inShape[3] + (inShape[3] - 1) * (upsamplingFactorVector[0] - 1) + padLVector[0] + padRVector[0],
    };

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());

    return mlir::success();
}
