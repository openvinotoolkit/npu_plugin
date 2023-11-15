//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/subgraph_optimizer.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace vpux {
namespace VPU {
//
// StrategyManager
//

// Higher level strategy manager class
// Its current purpose is to globally assign strategies
// In future it may have methods for finding sub-graphs
// and other strategy related utilities
class StrategyManager final {
public:
    explicit StrategyManager(mlir::func::FuncOp func, Logger log);

public:
    void assignMultiClusterStrategy(bool enableMultiClusterForSWLayer);
    void optimizeMulticlusterStrategy();
    void removeTemporaryMulticlusterStrategy();

private:
    mlir::func::FuncOp _func;
    Logger _log;
    LayerCostModel _costModel;
    SubgraphOptimizer _optimizer;
};
}  // namespace VPU
}  // namespace vpux
