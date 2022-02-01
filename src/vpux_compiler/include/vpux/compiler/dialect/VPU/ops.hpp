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

//
// Tiling
//

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp op, const TilingInfo& inputTiling) {
    VPUX_THROW_UNLESS(inputTiling.pads.hasValue(), "Missing tile information for paddings");

    auto newPadAttr = getPaddingAttr(op->getContext(), inputTiling.pads.getValue());

    op->padAttr(newPadAttr);
}

// Adjust bias attribute for specific output tile
template <typename ConcreteOp>
void adjustBias(ConcreteOp op, const TileInfo& outputTile) {
    auto biasOffset = Shape(outputTile.offsets);
    auto biasShape = Shape(outputTile.shape);
    for (const auto axis : {Dims4D::Act::N, Dims4D::Act::H, Dims4D::Act::W}) {
        biasOffset[axis] = 0;
        biasShape[axis] = 1;
    }

    const auto fullBiasAttr = op->biasAttr();
    auto newBiasAttr = fullBiasAttr.subview(biasOffset, biasShape);
    op->biasAttr(newBiasAttr);
}

// Adjust rawFilterShape attribute for specific output tile
template <typename ConcreteOp>
void adjustRawFilterShape(ConcreteOp op, const TileInfo& outputTile) {
    auto newRawFilterShape = Shape(parseIntArrayAttr<int64_t>(op->rawFilterShapeAttr()));

    newRawFilterShape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    op->rawFilterShapeAttr(getIntArrayAttr(op->getContext(), newRawFilterShape));
}

//
// Misc
//

void print(mlir::OpAsmPrinter& p, VPU::NCEClusterTilingOp op);
mlir::ParseResult parseNCEClusterTilingOp(mlir::OpAsmParser& parser, mlir::OperationState& result);

}  // namespace VPU
}  // namespace vpux
