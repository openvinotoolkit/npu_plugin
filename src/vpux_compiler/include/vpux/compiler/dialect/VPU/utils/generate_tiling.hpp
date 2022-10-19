//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

namespace vpux {
namespace VPU {

// Experimental number to avoid memory fragmentation when generating tiling.
static constexpr double FRAGMENTATION_AVOID_RATIO = 0.9;

// Experimental number to define large constant size
// The constant filter is considered as large constant value
// when its size is bigger than CMXSize*LARGE_CONST_THRESHOLD_RATIO
static constexpr double LARGE_CONST_THRESHOLD_RATIO = 0.25;

// An experimental number from MCM activation prefetch pass.
// The purpose is to avoid excessive tiling.
static constexpr int MAX_PREFETCH_TILING_TIME = 3;

OutputTiling getTilingStrategy(mlir::Operation* op, Logger log, TilingMode tilingMode = TilingMode::ISOLATED);
mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log);
mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, OutputTiling tiles,
                                      mlir::PatternRewriter& rewriter, Logger log);
mlir::Operation* getParentTargetOp(mlir::Operation* op);
bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log);
bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log);

bool archSupportsSwLayerTiling(VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
