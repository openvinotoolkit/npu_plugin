//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantize.input().getType().cast<mlir::ShapedType>();

    SmallVector<int64_t> scalarShape{1};
    auto ui8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    inferredReturnShapes.emplace_back(inType.getShape(), ui8Type);
    inferredReturnShapes.emplace_back(scalarShape, inType.getElementType());
    inferredReturnShapes.emplace_back(scalarShape, ui8Type);

    return mlir::success();
}
