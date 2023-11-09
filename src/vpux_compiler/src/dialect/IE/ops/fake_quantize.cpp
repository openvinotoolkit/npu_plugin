//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FakeQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FakeQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.input().getType().cast<mlir::ShapedType>();
    const auto inputLowType = quantize.input_low().getType().cast<mlir::ShapedType>();
    const auto inputHighType = quantize.input_high().getType().cast<mlir::ShapedType>();
    const auto outputLowType = quantize.output_low().getType().cast<mlir::ShapedType>();
    const auto outputHighType = quantize.output_high().getType().cast<mlir::ShapedType>();
    const auto autob = quantize.auto_broadcast();

    const auto outShapeOrResult =
            IE::broadcastEltwiseShape({inputType.getShape(), inputLowType.getShape(), inputHighType.getShape(),
                                       outputLowType.getShape(), outputHighType.getShape()},
                                      autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        inferredReturnShapes.emplace_back(outShapeOrResult.value(), inputType.getElementType());
    }

    return outShapeOrResult;
}

mlir::OpFoldResult vpux::IE::FakeQuantizeOp::fold(ArrayRef<mlir::Attribute> /*operands*/) {
    if (auto fakeQuantize = input().getDefiningOp<IE::FakeQuantizeOp>()) {
        const auto cstMinInSecondFQ = input_low();
        const auto cstMaxInSecondFQ = input_high();
        const auto cstMinOutSecondFQ = output_low();
        const auto cstMaxOutSecondFQ = output_high();
        const auto cstMinInFirstFQ = fakeQuantize.input_low();
        const auto cstMaxInFirstFQ = fakeQuantize.input_high();
        const auto cstMinOutFirstFQ = fakeQuantize.output_low();
        const auto cstMaxOutFirstFQ = fakeQuantize.output_high();
        if (cstMinInSecondFQ == cstMinInFirstFQ && cstMaxInSecondFQ == cstMaxInFirstFQ &&
            cstMinOutSecondFQ == cstMinOutFirstFQ && cstMaxOutSecondFQ == cstMaxOutFirstFQ) {
            return input();
        }
    }

    return nullptr;
}
