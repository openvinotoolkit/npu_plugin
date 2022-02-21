//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>

namespace vpux {
namespace IE {

// Experimental number to avoid memory fragmentation when generating tiling.
static constexpr double FRAGMENTATION_AVOID_RATIO = 0.9;

// An experimental number from MCM activation prefetch pass.
// The purpose is to avoid excessive tiling.
static constexpr int MAX_PREFETCH_TILING_TIME = 3;

OutputTiling getTilingStrategy(mlir::Operation* op, Logger log, TilingMode tilingMode = TilingMode::ISOLATED_TILING);
mlir::Value reifyTile(IE::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log);
mlir::LogicalResult applyTileStrategy(IE::TilingBuilderOpInterface origOp, OutputTiling tiles,
                                      mlir::PatternRewriter& rewriter, Logger log);
mlir::Operation* getParentTargetOp(mlir::Operation* op);
bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log);

}  // namespace IE
}  // namespace vpux
