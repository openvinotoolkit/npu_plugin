//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FakeQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FakeQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.getInput().getType().cast<mlir::ShapedType>();
    const auto inputLowType = quantize.getInputLow().getType().cast<mlir::ShapedType>();
    const auto inputHighType = quantize.getInputHigh().getType().cast<mlir::ShapedType>();
    const auto outputLowType = quantize.getOutputLow().getType().cast<mlir::ShapedType>();
    const auto outputHighType = quantize.getOutputHigh().getType().cast<mlir::ShapedType>();
    const auto autob = quantize.getAutoBroadcast();

    const auto outShapeOrResult =
            IE::broadcastEltwiseShape({inputType.getShape(), inputLowType.getShape(), inputHighType.getShape(),
                                       outputLowType.getShape(), outputHighType.getShape()},
                                      autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        inferredReturnShapes.emplace_back(outShapeOrResult.value(), inputType.getElementType());
    }

    return outShapeOrResult;
}

mlir::OpFoldResult vpux::IE::FakeQuantizeOp::fold(FoldAdaptor) {
    if (auto fakeQuantize = getInput().getDefiningOp<IE::FakeQuantizeOp>()) {
        const auto cstMinInSecondFQ = getInputLow();
        const auto cstMaxInSecondFQ = getInputHigh();
        const auto cstMinOutSecondFQ = getOutputLow();
        const auto cstMaxOutSecondFQ = getOutputHigh();
        const auto cstMinInFirstFQ = fakeQuantize.getInputLow();
        const auto cstMaxInFirstFQ = fakeQuantize.getInputHigh();
        const auto cstMinOutFirstFQ = fakeQuantize.getOutputLow();
        const auto cstMaxOutFirstFQ = fakeQuantize.getOutputHigh();
        if (cstMinInSecondFQ == cstMinInFirstFQ && cstMaxInSecondFQ == cstMaxInFirstFQ &&
            cstMinOutSecondFQ == cstMinOutFirstFQ && cstMaxOutSecondFQ == cstMaxOutFirstFQ) {
            return getInput();
        }
    }

    return nullptr;
}
