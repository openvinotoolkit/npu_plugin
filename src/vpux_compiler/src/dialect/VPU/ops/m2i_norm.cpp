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

bool vpux::VPU::M2INormOp::fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    Byte requiredCMX(0);
    requiredCMX += input.getTotalAllocSize();
    requiredCMX += output.getTotalAllocSize();
    return requiredCMX <= getTotalCMXSize(op);
}

//
// isSupported
//

bool vpux::VPU::M2INormOp::isSupported(IE::BatchNormInferenceOp op, LogCb logCb) {
    const auto inType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto outType = op.output().getType().cast<vpux::NDTypeInterface>();

    // Norm only defined for FP, and M2I only supports fp16
    auto iType = inType.getElementType();
    auto oType = outType.getElementType();

    if (!iType.isF16()) {
        logCb(llvm::formatv("Op only supports F16 input, got {0}", iType));
        return false;
    }

    if (!oType.isF16()) {
        logCb(llvm::formatv("Op only supports F16 output, got {0}", oType));
        return false;
    }

    if (!fitIntoCMX(op, inType, outType)) {
        logCb(llvm::formatv("Op doesn't fit into CMX memory"));
        return false;
    }

    const auto rank = inType.getShape().size();
    if (rank != 4) {
        logCb(llvm::formatv("Op only supports 4D shape, got {0}", rank));
        return false;
    }

    const auto lnStride = getM2iLineStride(inType, Dims4D::Act::W.ind());
    if (!VPU::isM2iLineStrideSupported(lnStride)) {
        logCb(llvm::formatv("Convert to M2I : line-stride NOT multiple of 16, got {0}", lnStride));
        return false;
    }

    return true;
}

//
// inferReturnTypes
//

mlir::LogicalResult vpux::VPU::M2INormOp::inferReturnTypes(mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::RegionRange,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    M2INormOpAdaptor op(operands, attrs);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = op.input().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

//
// EMU serialize
//

EMU::BlobWriter::SpecificTask vpux::VPU::M2INormOp::serialize(EMU::BlobWriter& /*writer*/) {
    VPUX_THROW("M2INormOp lowers to M2ITaskOp");
}
