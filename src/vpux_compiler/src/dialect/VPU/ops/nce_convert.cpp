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

namespace {

bool isTensorsSupported(mlir::Operation* op, VPU::NCEInvariant::LogCb logCb) {
    const auto input = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto output = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

    if (input.getRank() != 4 || output.getRank() != 4) {
        logCb(llvm::formatv("Only 4D tensors are supported"));
        return false;
    }

    const auto inputOrder = input.getDimsOrder();
    const auto outputOrder = output.getDimsOrder();

    if (inputOrder != DimsOrder::NHWC || outputOrder != DimsOrder::NHWC) {
        logCb(llvm::formatv("Unsupported layout"));
        return false;
    }

    return true;
}

};

//
// fitIntoCMX
//

bool vpux::VPU::NCEConvertOp::fitIntoCMX(vpux::NDTypeInterface input,
                                         vpux::NDTypeInterface output) {
    Byte requiredCMX(0);

    for (const auto& type : {input, input, output}) {
        requiredCMX += type.getTotalAllocSize();
    }

    return requiredCMX <= getTotalCMXSize(getOperation());
}

//
// isSupported
//

template<class ConcreteOp>
bool vpux::VPU::NCEConvertOp::isSupported(ConcreteOp op, VPU::ArchKind arch, vpux::VPU::NCEInvariant::LogCb logCb) {
    return false;
}

template<>
bool vpux::VPU::NCEConvertOp::isSupported(IE::DequantizeOp dequantizeOp, VPU::ArchKind, vpux::VPU::NCEInvariant::LogCb logCb) {
    auto inElemType = dequantizeOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    return !inElemType.isa<mlir::quant::UniformQuantizedPerAxisType>() && isTensorsSupported(dequantizeOp, logCb);
}

template<>
bool vpux::VPU::NCEConvertOp::isSupported(IE::QuantizeOp quantizeOp, VPU::ArchKind arch, vpux::VPU::NCEInvariant::LogCb logCb) {
    if (!isTensorsSupported(quantizeOp, logCb)) {
        return false;
    }

    auto outType = quantizeOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto isPerChannelQuantized = outType.getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
    const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(arch, outType);

    auto outputLayerUsers = quantizeOp.output().getUsers();
    auto anyUserIsConv = !outputLayerUsers.empty() && ::llvm::any_of(outputLayerUsers, [](auto user) {
        return mlir::isa<IE::ConvolutionOp>(user);
    });

    return !(anyUserIsConv && canUseCMajor) && !isPerChannelQuantized;
}

//
// InferShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    NCEConvertOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inShape = getShape(op.input());
    // TODO: Copied from EltWiseOp, so should work, but looks strange. Need be double checked
    const auto inElemType = op.input().getType().cast<vpux::NDTypeInterface>().getElementType();

    inferredReturnShapes.emplace_back(inShape.raw(), inElemType);
    return mlir::success();
}

//
// LayoutInfoOpInterface
//

void vpux::VPU::NCEConvertOp::inferLayoutInfo(IE::LayerLayoutInfo& info) {
    info.fill(DimsOrder::NHWC);
}
