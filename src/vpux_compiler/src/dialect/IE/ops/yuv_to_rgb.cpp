//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::YuvToRgbOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::YuvToRgbOpAdaptor colorConv(operands, attrs);
    if (mlir::failed(colorConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = colorConv.input1().getType().cast<mlir::ShapedType>();
    const auto shape = inType.getShape();
    if (shape[3] != 1) {
        return errorAt(loc, "Incorrect input shape format: '{0}'", shape);
    }

    SmallVector<int64_t> outShape{shape[0], shape[1], shape[2], 3};

    if (colorConv.input2() == nullptr) {
        VPUX_THROW_UNLESS(colorConv.input3() == nullptr, "1xPlane config error");
        VPUX_THROW_UNLESS(((outShape[1] * 2) % 3) == 0, "Invalid height");
        outShape[1] = outShape[1] * 2 / 3;
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
