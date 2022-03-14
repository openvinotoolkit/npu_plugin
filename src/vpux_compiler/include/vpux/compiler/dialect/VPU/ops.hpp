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

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
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

mlir::LogicalResult verifyOp(NCEClusterTilingOp op);
mlir::LogicalResult verifyOp(YieldOp op);

mlir::LogicalResult verifyOp(DistributedCastOp op);

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

mlir::LogicalResult sameOrder(mlir::Location loc, VPU::DistributedTensorType inDistributedType,
                              VPU::DistributedTensorType outDistributedType);
mlir::LogicalResult sameOrder(mlir::Location loc, VPUIP::DistributedBufferType inDistributedType,
                              VPUIP::DistributedBufferType outDistributedType);

template <typename T, enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                      std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                  bool> = true>
mlir::LogicalResult isDistributedCastCompatible(T inDistributedType, T outDistributedType) {
    const auto loc = mlir::UnknownLoc::get(inDistributedType.getContext());

    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        return errorAt(loc, "Mismatch between shapes for input ({0}) and output ({1}).", inDistributedType.getShape(),
                       outDistributedType.getShape());
    }

    if (inDistributedType.getElementType() != outDistributedType.getElementType()) {
        return errorAt(loc, "Mismatch between element types for input ({0}) and output ({1}).",
                       inDistributedType.getElementType(), outDistributedType.getElementType());
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        return errorAt(loc, "Mismatch between memspaces for input ({0}) and output ({1}).",
                       inDistributedType.getMemSpace(), outDistributedType.getMemSpace());
    }

    const auto sameOrderCheck = sameOrder(loc, inDistributedType, outDistributedType);
    if (sameOrderCheck.failed()) {
        return sameOrderCheck;
    }

    const auto inDistributionAttr = inDistributedType.getDistribution();
    const auto outDistributionAttr = outDistributedType.getDistribution();

    if (inDistributionAttr.num_clusters() != outDistributionAttr.num_clusters()) {
        return errorAt(loc, "Mismatch between number of clusters for input ({0}) and output ({1}).",
                       inDistributionAttr.num_clusters(), outDistributionAttr.num_clusters());
    }

    if (inDistributionAttr.alignment() != outDistributionAttr.alignment()) {
        return errorAt(loc, "Mismatch between tensor alignment of clusters for input ({0}) and output ({1}).",
                       inDistributionAttr.alignment(), outDistributionAttr.alignment());
    }

    const auto inDistributionMode = inDistributionAttr.mode().getValue();
    const auto outDistributionMode = outDistributionAttr.mode().getValue();

    if (inDistributionMode != outDistributionMode) {
        if (VPU::areDistributionModesCompatible(inDistributionMode, outDistributionMode).failed()) {
            return errorAt(loc, "Incompatible distribution modes for input ({0}) and output ({1}).",
                           VPU::stringifyDistributionMode(inDistributionMode),
                           VPU::stringifyDistributionMode(outDistributionMode));
        }
    }

    return mlir::success();
}

}  // namespace VPU
}  // namespace vpux
