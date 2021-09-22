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

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::QuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantize.input().getType().cast<mlir::ShapedType>();
    const auto dstElemType = quantize.dstElemType().getValue();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

//
// fold
//

namespace {

mlir::quant::QuantizedType extractQuantizedType(mlir::Value operand) {
    const auto elemType = operand.getType().cast<mlir::ShapedType>().getElementType();
    const auto quantType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Type must be quantized, but provided {0}", elemType);
    return quantType;
};

}  // namespace

mlir::OpFoldResult vpux::IE::QuantizeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto quantType = extractQuantizedType(output());
        const auto quantStorageType = normalizeQuantStorageType(quantType);
        return cst.convertElemType(quantStorageType).quantCast(quantType);
    }

    if (auto dequantize = input().getDefiningOp<IE::DequantizeOp>()) {
        if (dequantize.input().getType() == output().getType()) {
            return dequantize.input();
        }
    }

    return nullptr;
}
