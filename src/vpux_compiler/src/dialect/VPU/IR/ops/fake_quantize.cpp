//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FakeQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FakeQuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputLowType = quantize.getInputLow().getType().cast<vpux::NDTypeInterface>();
    const auto inputHighType = quantize.getInputHigh().getType().cast<vpux::NDTypeInterface>();
    const auto outputLowType = quantize.getOutputLow().getType().cast<vpux::NDTypeInterface>();
    const auto outputHighType = quantize.getOutputHigh().getType().cast<vpux::NDTypeInterface>();
    const auto autob = quantize.getAutoBroadcast();

    const auto outShapeOrResult = IE::broadcastEltwiseShape(
            {inputType.getShape().raw(), inputLowType.getShape().raw(), inputHighType.getShape().raw(),
             outputLowType.getShape().raw(), outputHighType.getShape().raw()},
            autob, loc);

    if (mlir::succeeded(outShapeOrResult)) {
        const auto outType = inputType.changeShape(Shape(outShapeOrResult.value()));
        inferredReturnTypes.push_back(outType);
    }

    return outShapeOrResult;
}
