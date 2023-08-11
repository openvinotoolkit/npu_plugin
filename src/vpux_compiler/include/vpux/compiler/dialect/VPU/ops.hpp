//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

bool isVFNCESupported(VPU::NCEOpInterface op);

mlir::LogicalResult sameLayout(VPU::DistributedTensorType inDistributedType,
                               VPU::DistributedTensorType outDistributedType, LogCb logCb = emptyLogCb);
mlir::LogicalResult sameLayout(VPUIP::DistributedBufferType inDistributedType,
                               VPUIP::DistributedBufferType outDistributedType, LogCb logCb = emptyLogCb);

template <typename T, enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                      std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                  bool> = true>
mlir::LogicalResult areDistributionAttrsCompatible(T sourceType, T targetType,
                                                   const bool allowOverlapWithDiffConfig = false) {
    const auto sourceAttr = sourceType.getDistribution();
    const auto targetAttr = targetType.getDistribution();

    const auto inDistributionMode = sourceAttr.mode().getValue();
    const auto outDistributionMode = targetAttr.mode().getValue();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::areDistributionModesCompatible(inDistributionMode, outDistributionMode).failed()) {
            return mlir::failure();
        }
    }

    // Check if the distributed tensor has the full tensor on each cluster
    auto isMemoryFullSizeMode = [&](VPU::DistributionMode mode) -> bool {
        return VPU::bitEnumContains(mode, VPU::DistributionMode::DUPLICATED) ||
               VPU::bitEnumContains(mode, VPU::DistributionMode::MULTICASTED);
    };

    // Only check the alignment when the tensor needs to split
    // For FullSizeTensor, e.g., DUPLICATED and MULTICASTED, tensors might be compatible even though
    // they have different alignment attributes. Because the tensors are aligned and the same on each cluster
    // For tensors that need to split, the same alignments are required to make sure tensors compatible on each cluster
    if (!(isMemoryFullSizeMode(inDistributionMode) && isMemoryFullSizeMode(outDistributionMode)) &&
        sourceAttr.alignment() != targetAttr.alignment()) {
        return mlir::failure();
    }

    const auto inDistributionNumClusters = sourceAttr.num_clusters();
    const auto outDistributionNumClusters = targetAttr.num_clusters();

    if (VPU::areDistributionNumClustersCompatible(inDistributionNumClusters, outDistributionNumClusters).failed()) {
        return mlir::failure();
    }

    // Ensure the memory view for the source and target distributions are the same,
    // no matter the attributes of the distribution.
    // For example, given:
    // sourceAttr = SEGMENTED across 2 clusters without uniformDistributedSegments
    // targetAttr = SEGMENTED across 2 clusters with uniformDistributedSegments
    // memory view will always be the same, so the distribution attrs are compatible.
    auto arePerClusterMemoryShapeAndOffsetsEqual = [&]() -> bool {
        auto srcMemoryOffsets = sourceType.getPerClusterMemoryShapeOffsets();
        auto targetMemoryOffsets = targetType.getPerClusterMemoryShapeOffsets();

        auto srcMemoryShapes = sourceType.getPerClusterMemoryShapes();
        auto targetMemoryShapes = targetType.getPerClusterMemoryShapes();

        return (srcMemoryOffsets == targetMemoryOffsets) && (srcMemoryShapes == targetMemoryShapes);
    };

    if ((inDistributionMode == VPU::DistributionMode::SEGMENTED) &&
        (outDistributionMode == VPU::DistributionMode::SEGMENTED)) {
        const auto inDistributionNumTiles = parseIntArrayAttr<int64_t>(sourceAttr.num_tiles());
        const auto outDistributionNumTiles = parseIntArrayAttr<int64_t>(targetAttr.num_tiles());
        if (inDistributionNumTiles != outDistributionNumTiles) {
            return mlir::failure();
        }

        return arePerClusterMemoryShapeAndOffsetsEqual() ? mlir::success() : mlir::failure();
    }

    if ((inDistributionMode == VPU::DistributionMode::OVERLAPPED) &&
        (outDistributionMode == VPU::DistributionMode::OVERLAPPED)) {
        const auto inDistributionNumTiles = parseIntArrayAttr<int64_t>(sourceAttr.num_tiles());
        const auto outDistributionNumTiles = parseIntArrayAttr<int64_t>(targetAttr.num_tiles());
        if (inDistributionNumTiles != outDistributionNumTiles) {
            return mlir::failure();
        }

        if (allowOverlapWithDiffConfig) {
            return mlir::success();
        }

        return arePerClusterMemoryShapeAndOffsetsEqual() ? mlir::success() : mlir::failure();
    }

    return mlir::success();
}

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

    const auto sameLayoutCheck = sameLayout(inDistributedType, outDistributedType, logCb);
    if (sameLayoutCheck.failed()) {
        return mlir::failure();
    }

    if (areDistributionAttrsCompatible(inDistributedType, outDistributedType).failed()) {
        logCb(formatv("Mismatch between distributionAttr for input ({0}) and output ({1}).",
                      inDistributedType.getDistribution(), outDistributedType.getDistribution()));
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
