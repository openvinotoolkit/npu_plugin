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

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(mlir::ShapedType input1, mlir::ShapedType input2, mlir::ShapedType output) {
    Byte requiredCMX(0);

    for (const auto& type : {input1, input2, output}) {
        requiredCMX += getTotalSize(type);
    }

    return requiredCMX <= getTotalCMXSize(getOperation());
}

//
// isSupported
//

bool vpux::VPU::NCEEltwiseOp::isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                          NCEInvariant::LogCb logCb) {
    const auto input1 = op->getOperand(0).getType().cast<mlir::ShapedType>();
    const auto input2 = op->getOperand(1).getType().cast<mlir::ShapedType>();
    const auto output = op->getResult(0).getType().cast<mlir::ShapedType>();

    if (input1.getRank() != 4 || input2.getRank() != 4 || output.getRank() != 4) {
        logCb(llvm::formatv("Only 4D tensors are supported"));
        return false;
    }

    if (input1.getShape() != input2.getShape()) {
        logCb(llvm::formatv("Broadcasting is not supported"));
        return false;
    }

    // Output type can differ from input type. In case of quantization this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    const auto input1ElemType = input1.getElementType();
    const auto input2ElemType = input2.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (input1ElemType != input2ElemType) {
            return false;
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        auto qInput1 = input1ElemType.cast<mlir::quant::UniformQuantizedType>();
        auto qInput2 = input2ElemType.cast<mlir::quant::UniformQuantizedType>();

        if (qInput1.getExpressedType() != qInput2.getExpressedType() ||
            qInput1.getStorageType() != qInput2.getStorageType() || qInput1.isSigned() != qInput2.isSigned()) {
            logCb(llvm::formatv("Mismatch in inputs quantization parameters"));
            return false;
        }

        if (!allowDifferentZp && qInput1.getZeroPoint() != qInput2.getZeroPoint()) {
            logCb(llvm::formatv("Mismatch in inputs zero points"));
            return false;
        }

        if (!allowDifferentScales && qInput1.getScale() != qInput2.getScale()) {
            logCb(llvm::formatv("Mismatch in inputs quantization scales"));
            return false;
        }
    } else {
        logCb(llvm::formatv("Unsupported inputs element types"));
        return false;
    }

    if (!NCEInvariant::isActTypeSupported(input1, getInputChannelAlignment(input1)) ||
        !NCEInvariant::isActTypeSupported(input2, getInputChannelAlignment(input2)) ||
        !NCEInvariant::isActTypeSupported(output, getOutputChannelAlignment(output))) {
        logCb(llvm::formatv("Misaligned tensor shape"));
        return false;
    }

    const auto inputOrder1 = DimsOrder::fromType(input1);
    const auto inputOrder2 = DimsOrder::fromType(input2);
    const auto outputOrder = DimsOrder::fromType(output);

    if (inputOrder1 != DimsOrder::NHWC || inputOrder2 != DimsOrder::NHWC || outputOrder != DimsOrder::NHWC) {
        logCb(llvm::formatv("Unsupported layout"));
        return false;
    }

    return true;
}

//
// InferShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCEEltwiseOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto shape1 = getShape(op.input1());
    const auto shape2 = getShape(op.input2());

    if (shape1 != shape2) {
        return errorAt(loc, "Broadcasting is not supported for {0} operation", NCEEltwiseOp::getOperationName());
    }

    const auto elemType1 = op.input1().getType().cast<mlir::ShapedType>().getElementType();

    inferredReturnShapes.emplace_back(shape1.raw(), elemType1);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCEEltwiseOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.fill(DimsOrder::NHWC);
}
