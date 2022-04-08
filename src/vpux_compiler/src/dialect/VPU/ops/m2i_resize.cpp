//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/m2i_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::M2IResizeOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input,
                                        vpux::NDTypeInterface output) {
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(
                   getArch(op), {input.getTotalAllocSize(), output.getTotalAllocSize()}) <= getTotalCMXSize(op);
}

//
// isSupported
//

bool vpux::VPU::M2IResizeOp::isSupported(IE::InterpolateOp op, LogCb logCb) {
    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.output().getType().cast<vpux::NDTypeInterface>();

    const auto iType = inType.getElementType();
    if (!(iType.isUnsignedInteger(8) || iType.isF16())) {
        logCb(llvm::formatv("Op only supports UI8/F16 input, got {0}", iType));
        return false;
    }
    const auto oType = outType.getElementType();
    if (!(oType.isUnsignedInteger(8) || oType.isF16())) {
        logCb(llvm::formatv("Op only supports UI8/F16 output, got {0}", oType));
        return false;
    }

    if (!fitIntoCMX(op, inType, outType)) {
        logCb(llvm::formatv("Op doesn't fit into CMX memory"));
        return false;
    }

    const auto shapeCalcMode = op.attr().shape_calc_mode().getValue();
    if (shapeCalcMode != IE::InterpolateCalcMode::SIZES) {
        logCb(llvm::formatv("Op only implements 'sizes' mode, got {0}", shapeCalcMode));
        return false;
    }

    const auto sizesSize = op.sizes_attrAttr().size();
    const auto axesSize = op.axes_attrAttr().size();
    if (sizesSize != 2) {
        logCb(llvm::formatv("M2I can only resize 2D images, got {0}D", sizesSize));
        return false;
    }

    if (sizesSize != axesSize) {
        logCb(llvm::formatv("Interpolate sizes/axes attr must have same size, got {0}, {1}", sizesSize, axesSize));
        return false;
    }

    const auto axes = parseIntArrayAttr<int64_t>(op.axes_attrAttr());
    const auto Waxis = axes[1];  // H(0),W(1)
    const auto iStride = getM2iLineStride(inType, Waxis);
    if (!VPU::isM2iLineStrideSupported(iStride)) {
        logCb(llvm::formatv("Input line-stride NOT multiple of 16, got {0}", iStride));
        return false;
    }
    const auto oStride = getM2iLineStride(outType, Waxis);
    if (!VPU::isM2iLineStrideSupported(oStride)) {
        logCb(llvm::formatv("Output line-stride NOT multiple of 16, got {0}", oStride));
        return false;
    }
    // Check consecutive axes
    if (axes[0] != (axes[1] - 1)) {
        logCb(llvm::formatv("Axes need to be consecutive values, got {0}, {1}", axes[0], axes[1]));
        return false;
    }

    const auto interpMode = op.attr().mode().getValue();
    if (!((interpMode == IE::InterpolateMode::NEAREST) || (interpMode == IE::InterpolateMode::LINEAR))) {
        logCb(llvm::formatv("Op only supports nearest/linear interpolate, got {0}", interpMode));
        return false;
    }

    // Interleaved fp16 not supported by M2I-HW
    const auto lastAxis = axes[axesSize - 1];
    const auto lastDim = outType.getRank() - 1;
    if (oType.isF16() && (lastAxis != lastDim)) {
        logCb(llvm::formatv("Interleaved fp16 not supported by M2I-HW, expecting last_axis == last_dim, got {0} != {1}",
                            lastAxis, lastDim));
        return false;
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(op.attr().pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.attr().pads_end());

    auto isNotZero = [](auto val) {
        return val != 0;
    };

    if (llvm::any_of(padsBegin, isNotZero) || llvm::any_of(padsEnd, isNotZero)) {
        logCb(llvm::formatv("Op does not support pads"));
        return false;
    }

    return true;
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2IResizeOp::inferReturnTypes(mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::RegionRange,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    M2IResizeOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    // Note: limited to 'shape_calculation_mode = sizes'
    const auto outSize = parseIntArrayAttr<int64_t>(op.sizes());
    const auto outAxes = parseIntArrayAttr<int64_t>(op.axes());

    SmallVector<int64_t> outShape;

    for (size_t i = 0; i < inShape.size(); i++) {
        outShape.emplace_back(inShape[i]);
    }

    // Patch dims
    if (outSize.size() != outAxes.size()) {
        VPUX_THROW("Sizes and Axes vectors must have same size !");
    }
    for (size_t i = 0; i < outAxes.size(); i++) {
        outShape[outAxes[i]] = outSize[i];
    }

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// EMU serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::M2IResizeOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("M2IResizeOp lowers to M2ITaskOp");
}
