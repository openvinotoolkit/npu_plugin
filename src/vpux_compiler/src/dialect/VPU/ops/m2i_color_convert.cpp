//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/m2i_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::M2IColorConvertOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input,
                                              vpux::NDTypeInterface output) {
    // Note: for 1xPlane config, 1st input fully dictates the size
    Byte requiredCMX(0);
    requiredCMX += input.getTotalAllocSize();
    requiredCMX += output.getTotalAllocSize();
    return requiredCMX <= getTotalCMXSize(op);
}

//
// isSupported
//

bool vpux::VPU::M2IColorConvertOp::isSupported(IE::YuvToRgbOp op, LogCb logCb) {
    const auto inType = op.input1().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.output().getType().cast<vpux::NDTypeInterface>();

    if (!fitIntoCMX(op, inType, outType)) {
        logCb(llvm::formatv("Op doesn't fit into CMX memory"));
        return false;
    }

    if ((op.input2() != nullptr) || (op.input3() != nullptr)) {
        logCb(llvm::formatv("Convert to M2I : only single-plane supported for now, got {0}", op.getNumOperands()));
        return false;
    }

    // M2I only supports UI8 for NV12/I420. Other modes could be enabled
    if (!(inType.getElementType().isUnsignedInteger(8) &&
          ((op.inFmt() == IE::ColorFmt::NV12) || (op.inFmt() == IE::ColorFmt::I420)))) {
        logCb(llvm::formatv("Convert to M2I : unsupported {0} in format and type {1}", op.inFmt(), inType));
        return false;
    }

    const auto lnStride = getM2iLineStride(inType, 2);  // NHW(2)C
    if (!VPU::isM2iLineStrideSupported(lnStride)) {
        logCb(llvm::formatv("Convert to M2I : line-stride NOT multiple of 16, got {0}", lnStride));
        return false;
    }

    return true;
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2IColorConvertOp::inferReturnTypes(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    M2IColorConvertOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    // Y,UV (1 or 2 plane configs) are exected to have C = 1
    if (inShape[3] != 1) {
        return errorAt(loc, "Incorrect number of channels: expecting 1, got '{0}'", inShape[3]);
    }

    // OK for NV12/I420 -> RGB/BGR
    SmallVector<int64_t> outShape{inShape[0], inShape[1], inShape[2], 3};
    // input Height is big enough to include Chroma, so lower for RGB output
    outShape[1] = outShape[1] * 2 / 3;

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// EMU serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::M2IColorConvertOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("M2IColorConvertOp lowers to M2ITaskOp");
}
