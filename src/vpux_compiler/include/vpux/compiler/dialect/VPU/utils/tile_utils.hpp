//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

namespace vpux {
namespace VPU {

// Convolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::ConvolutionOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEConvolutionOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCECompressConvolutionOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEMaxPoolOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);
// AveragePool

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEAveragePoolOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);

// GroupConvolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::GroupConvolutionOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);
SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::NCEDepthConvolutionOp origOp, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);

SmallVector<vpux::NDTypeInterface> getTileTypes(mlir::Operation* op, const TileInfo& outTile,
                                                const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::NCEDepthConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::MaxPoolOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::NCEMaxPoolOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::NCEAveragePoolOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::MaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getEltwiseRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling,
                           const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::AddOp op, const vpux::TileInfo& tiling, const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::AddOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);
Byte getRequiredCMX(VPU::MultiplyOp op, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::MultiplyOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::SubtractOp op, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::SubtractOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::AndOp op, const vpux::TileInfo& tiling, const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::AndOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXForWeight(mlir::Operation* op, const vpux::TileInfo& tiling,
                             const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling, Logger log,
                    const mlir::Optional<InputTiling>& inputTiles = None);

Byte getRequiredCMXSize(ArrayRef<vpux::NDTypeInterface> operands);

Byte getRequiredCMXSizeForNCEOps(ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels);

Byte getRequiredCMXSizeForDefaultOps(mlir::Operation* op);
}  // namespace VPU
}  // namespace vpux
