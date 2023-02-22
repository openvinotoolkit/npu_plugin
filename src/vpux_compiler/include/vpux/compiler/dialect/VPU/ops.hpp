//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/EMU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/dialect/VPUIP/types.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/generated/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace VPU {

mlir::LogicalResult verifyConv(mlir::Location loc, ArchKind arch, NCEConvolutionOpAdaptor op, mlir::Value output);

mlir::LogicalResult verifyOp(NCEConvolutionOp op);
mlir::LogicalResult verifyOp(NCEDepthConvolutionOp op);
mlir::LogicalResult verifyOp(NCEMaxPoolOp op);
mlir::LogicalResult verifyOp(NCEAveragePoolOp op);
mlir::LogicalResult verifyOp(NCEPermuteQuantizeOp op);
mlir::LogicalResult verifyOp(BucketizeOp op);
mlir::LogicalResult verifyOp(PReluOp op);
mlir::LogicalResult verifyOp(GatherNDOp op);

mlir::LogicalResult verifyOp(NCEClusterTilingOp op);
mlir::LogicalResult verifyOp(YieldOp op);

mlir::LogicalResult verifyOp(DistributedCastOp op);
mlir::LogicalResult verifyOp(StorageElementTableOp op);
mlir::LogicalResult verifyOp(LayoutCastOp op);
mlir::LogicalResult verifyOp(ConcatOp op);

//
// Tiling
//

// Returns a WeightsTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getWeightsTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile) {
    const auto origWeightsTable = origOp->weightsTable();
    VPUX_THROW_UNLESS(origWeightsTable != nullptr, "The operation {0} doesn't have a WeightsTable", *origOp);

    const auto origWeightsTableShape = getShape(origWeightsTable);
    VPUX_THROW_UNLESS(origWeightsTableShape[Dim(0)] == getShape(origOp->output())[Dims4D::Act::C] &&
                              origWeightsTableShape[Dim(1)] == 1 && origWeightsTableShape[Dim(2)] == 1 &&
                              origWeightsTableShape[Dim(3)] == VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected WeightsTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origWeightsTableShape, getShape(origOp->output()));

    // Each N-wise batch of the WeightsTable corresponds to its own output channel
    TileInfo weightsTableTile(origWeightsTableShape);
    weightsTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    weightsTableTile.shape[Dim(0)] = outputTile.shape[Dims4D::Act::C];
    return weightsTableTile;
}

// Returns an ActivationWindow tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getActivationWindowTile(ConcreteOp* origOp, const vpux::TileInfo& /*outputTile*/) {
    const auto origActivationWindow = origOp->activationWindow();
    VPUX_THROW_UNLESS(origActivationWindow != nullptr, "The operation {0} doesn't have an ActivationWindow", *origOp);

    const auto origActivationWindowShape = getShape(origActivationWindow);
    VPUX_THROW_UNLESS(origActivationWindowShape[Dim(0)] == 1 && origActivationWindowShape[Dim(1)] == 1 &&
                              origActivationWindowShape[Dim(2)] == 1,
                      "Unexpected ActivationWindow shape type or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origActivationWindowShape, getShape(origOp->output()));

    // All output channels use the same only-one string in the table, so we just copy the whole thing
    return TileInfo(origActivationWindowShape);
}

// Returns an getInstructionListTableTile tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getInstructionListTableTile(ConcreteOp* origOp, const vpux::TileInfo& /*outputTile*/) {
    const auto origInstructionListTable = origOp->instructionListTable();
    VPUX_THROW_UNLESS(origInstructionListTable != nullptr, "The operation {0} doesn't have an InstructionListTable",
                      *origOp);

    const auto origInstructionListTableShape = getShape(origInstructionListTable);
    VPUX_THROW_UNLESS(origInstructionListTableShape[Dim(0)] == 1 && origInstructionListTableShape[Dim(1)] == 1 &&
                              origInstructionListTableShape[Dim(2)] == 1,
                      "Unexpected InstructionListTable shape type or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origInstructionListTableShape, getShape(origOp->output()));

    // All output channels use the same only-one string in the table, so we just copy the whole thing
    return TileInfo(origInstructionListTableShape);
}

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    VPUX_THROW_UNLESS(inputTiling.pads.hasValue(), "Missing tile information for paddings");

    auto newPadAttr = getPaddingAttr(op->getContext(), inputTiling.pads.getValue());

    op->padAttr(newPadAttr);
}

// Adjust rawFilterShape attribute for specific output tile
template <typename ConcreteOp>
void adjustRawFilterShape(ConcreteOp* op, const TileInfo& outputTile) {
    auto newRawFilterShape = Shape(parseIntArrayAttr<int64_t>(op->rawFilterShape()));

    newRawFilterShape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    op->rawFilterShapeAttr(getIntArrayAttr(op->getContext(), newRawFilterShape));
}

//
// Misc
//

void print(mlir::OpAsmPrinter& p, VPU::NCEClusterTilingOp op);
mlir::ParseResult parseNCEClusterTilingOp(mlir::OpAsmParser& parser, mlir::OperationState& result);

mlir::LogicalResult sameOrder(VPU::DistributedTensorType inDistributedType,
                              VPU::DistributedTensorType outDistributedType, LogCb logCb = emptyLogCb);
mlir::LogicalResult sameOrder(VPUIP::DistributedBufferType inDistributedType,
                              VPUIP::DistributedBufferType outDistributedType, LogCb logCb = emptyLogCb);

template <typename T, enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                      std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                  bool> = true>
mlir::LogicalResult isDistributedCastCompatible(T inDistributedType, T outDistributedType, LogCb logCb = emptyLogCb) {
    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        logCb(formatv("Mismatch between shapes for input ({0}) and output ({1}).", inDistributedType.getShape(),
                      outDistributedType.getShape()));
        return mlir::failure();
    }

    if (areDistributionElementTypesCompatible(inDistributedType.getElementType(), outDistributedType.getElementType())
                .failed()) {
        logCb(formatv("Mismatch between element types for input ({0}) and output ({1}).",
                      inDistributedType.getElementType(), outDistributedType.getElementType()));
        return mlir::failure();
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        logCb(formatv("Mismatch between memspaces for input ({0}) and output ({1}).", inDistributedType.getMemSpace(),
                      outDistributedType.getMemSpace()));
        return mlir::failure();
    }

    const auto sameOrderCheck = sameOrder(inDistributedType, outDistributedType, logCb);
    if (sameOrderCheck.failed()) {
        return mlir::failure();
    }

    const auto inDistributionAttr = inDistributedType.getDistribution();
    const auto outDistributionAttr = outDistributedType.getDistribution();

    if (areDistributionAttrsCompatible(inDistributionAttr, outDistributionAttr).failed()) {
        logCb(formatv("Mismatch between distributionAttr for input ({0}) and output ({1}).", inDistributionAttr,
                      outDistributionAttr));
        return mlir::failure();
    }

    return mlir::success();
}

template <typename T>
T vpux::VPU::NCEClusterTilingOp::getInnerTaskOpOfType() {
    return mlir::dyn_cast<T>(&body().front().front());
}
}  // namespace VPU
}  // namespace vpux
