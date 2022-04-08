//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::GridSampleOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GridSampleOpAdaptor gridSample(operands, attrs);

    if (mlir::failed(gridSample.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gridSample.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    const auto gridType = gridSample.grid().getType().cast<mlir::ShapedType>();
    const auto gridShape = gridType.getShape();

    SmallVector<int64_t> outShape = {inShape[0], inShape[1], gridShape[1], gridShape[2]};

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// verifyOp
//

mlir::LogicalResult vpux::IE::verifyOp(GridSampleOp op) {
    // E#61161 GridSample only support specific case

    const auto alignCorner = op.align_cornersAttr() != nullptr;
    if (alignCorner != 1) {
        return errorAt(op, "Unsupported case align_corner {0}", alignCorner);
    }

    if (op.modeAttr() != nullptr) {
        const auto mode = op.modeAttr().getValue();
        if (mode != IE::GridSampleMode::BILINEAR) {
            return errorAt(op, "Unsupported case mode {0}", mode);
        }
    }

    if (op.padding_modeAttr() != nullptr) {
        const auto paddingMode = op.padding_modeAttr().getValue();
        if (paddingMode != IE::GridSamplePaddingMode::BORDER) {
            return errorAt(op, "Unsupported case mode {0}", paddingMode);
        }
    }

    return mlir::success();
}
