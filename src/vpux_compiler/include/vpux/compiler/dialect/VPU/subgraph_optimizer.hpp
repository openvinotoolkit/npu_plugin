//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <queue>
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"

namespace vpux {
namespace VPU {
//
// Subgraph optimizer
//
class SubgraphOptimizer final {
public:
    SubgraphOptimizer(mlir::FuncOp func, Logger log);
    void optimizeStrategyAvoidSpillingOnModel();

private:
    bool isValidStrategy(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy);
    bool isStrategySOKLike(VPU::NCEOpInterface op, bool allowHK = true);
    VPU::NCEOpInterface findSourceOperation(VPU::NCEOpInterface currentOp, bool isParent);
    LayerCostModel::SpillingCost getSpillingCostWithSourceOp(VPU::NCEOpInterface currentOp,
                                                             VPU::NCEOpInterface sourceOp,
                                                             VPU::MultiClusterStrategy currentStrategy, bool isParent);
    std::pair<VPU::MultiClusterStrategy, double> getBestInSOKLikeStrategies(VPU::NCEOpInterface op, bool isParent,
                                                                            bool allowHK = true);
    void optimizeStrategyAvoidSpillingOnSubgraph(VPU::NCEOpInterface op);

    std::queue<VPU::NCEOpInterface> parents;
    std::queue<VPU::NCEOpInterface> children;
    std::map<VPU::NCEOpInterface, VPU::MultiClusterStrategy> parentsToChange;
    std::map<VPU::NCEOpInterface, VPU::MultiClusterStrategy> childrenToChange;
    std::set<mlir::Operation*> processedOps;
    // Keeps a record of operations where spill cost has already being taken into account
    std::set<mlir::Operation*> opsWithSpillWriteCounted;
    mlir::FuncOp _func;
    Logger _log;
    LayerCostModel _layerCostModel;
};

}  // namespace VPU
}  // namespace vpux
