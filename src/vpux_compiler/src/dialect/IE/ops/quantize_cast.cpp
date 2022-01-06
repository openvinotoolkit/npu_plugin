//
// Copyright Intel Corporation.
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
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::QuantizeCastOpAdaptor quantizeCast(operands, attrs);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantizeCast.input().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = quantizeCast.dstElemType().getValue();
    const auto outDesc = IE::getTensorAttr(inType);

    auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>();
    const auto outputWidth = (quantizedOutput)
        ? quantizedOutput.getStorageTypeIntegralWidth()
        : quantizeCast.dstElemType().getValue().getIntOrFloatBitWidth();

    if (auto integerInput = inType.getElementType().dyn_cast<mlir::IntegerType>()) {
        const auto inputWidth = integerInput.getWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Integer input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else if (auto quantizedInput = inType.getElementType().dyn_cast<mlir::quant::QuantizedType>()) {
        const auto inputWidth = quantizedInput.getStorageTypeIntegralWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantized input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else {
        return errorAt(loc, "Unsupported combination of input and output element types: {0} -> {1}",
                       inType.getElementType(), dstElemType);
    }

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::QuantizeCastOp::fold(vpux::ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}
