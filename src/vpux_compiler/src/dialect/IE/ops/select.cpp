//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SelectOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SelectOpAdaptor select(operands, attrs);
    if (mlir::failed(select.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = select.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = select.input2().getType().cast<mlir::ShapedType>();
    const auto in3Type = select.input3().getType().cast<mlir::ShapedType>();

    const auto outShapeRes = IE::broadcastEltwiseShape({in1Type.getShape(), in2Type.getShape(), in3Type.getShape()},
                                                       select.auto_broadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.getValue(), in2Type.getElementType());
    }

    return outShapeRes;
}
