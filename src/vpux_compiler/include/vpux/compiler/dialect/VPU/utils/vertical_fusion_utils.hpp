//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/vertical_fusion_storage.hpp"

namespace vpux {
namespace VPU {

// for now there is restriction for number of operations
// which might need bigger input size
constexpr int64_t MAXIMUM_VF_LENGTH = 3;

// min length of tensor by tiled axis. It limits number of tiles
// which we may increase in order to fit in CMX
constexpr int64_t MINIMUM_LENGTH_TILING = 4;

// information about input and output tiles for operands and result
using VFOperationTiling = std::pair<InputTiling, TileInfo>;

// storage keeps connection between argument number of the block and biggest tile
// for parent operation of the block for each separate VF tile
using TilingStorage = VFContainer<size_t, TileInfo>;

// storage keeps connection between operation in the block and its information
// about input and output tiles for each VF tile
using TilingOperationStorage = VFContainer<mlir::Operation*, VFOperationTiling, llvm::less_second>;

// function gets tiling information from VF subgraph and builds tiling info going up
// to arguments of the block
// it returns tiles for parent operations of the block and connection between them and block arguments
// for each VF tile
TilingStorage restoreTilingRegions(VPU::VerticalFusionOp vfOp, Logger log,
                                   const TilingOperationStorage::UPtr& opStorage = nullptr);
// calculate tiling regions based on particular tiling strategy
mlir::FailureOr<TilingStorage> calculateTilingRegions(VPU::VerticalFusionOp vfOp, ArrayRef<int64_t> tilingStrategy,
                                                      Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage = nullptr);
// calculate tiling regions based on known output tiles for last operation in the block
mlir::FailureOr<TilingStorage> calculateTilingRegions(VPU::VerticalFusionOp vfOp, OutputTiling tiles, Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage = nullptr);
// calculate recursively tiling regions for the block starting from last operation and known output tiles for it
// function builds connection between block arguments and tiles
// in case TilingOperationStorage pointer was passed, it filles in connection between each operation and
// its input and output tiles
mlir::FailureOr<TilingStorage> calculateTilingRegions(VPU::TilingBuilderOpInterface tilingBuilderOp, OutputTiling tiles,
                                                      Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage = nullptr,
                                                      mlir::Optional<size_t> numTile = None);

// calculate limit for number of tiles for set of operations
int64_t getTilingLimit(Dim axis, ArrayRef<mlir::Operation*> operations);

}  // namespace VPU
}  // namespace vpux
