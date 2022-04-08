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
    VPU::MultiClusterStrategy getRollbackStrategy(VPU::NCEOpInterface op);
    VPU::MultiClusterStrategy getRollbackStrategy(VPU::ClusteredOpInterface op);

    bool isValidStrategy(VPU::ClusteredOpInterface op, VPU::MultiClusterStrategy strategy);
    bool isStrategySOHLike(VPU::ClusteredOpInterface op);
    bool isStrategySOKLike(VPU::ClusteredOpInterface op);
    VPU::MultiClusterStrategy getBestInSOKLikeStrategies(VPU::ClusteredOpInterface op);
    VPU::MultiClusterStrategy getBestInSOHLikeStrategies(VPU::ClusteredOpInterface op);
    double getInputSpillingRollbackCostToMultiClusterLayer(VPU::ClusteredOpInterface nceOp,
                                                           VPU::MultiClusterStrategy strategy);
    double getInputSpillingRollbackCostToMultiClusterLayer(VPU::ClusteredOpInterface nceOp, mlir::Value input,
                                                           VPU::MultiClusterStrategy strategy);
    double getOutputSpillingRollbackCostToMultiClusterLayer(VPU::ClusteredOpInterface nceOp,
                                                            VPU::MultiClusterStrategy strategy);
    void optimizeStrategyAvoidSpillingOnSubgraph(VPU::ClusteredOpInterface op);

    SmallVector<VPU::ClusteredOpInterface> layersNeedRollback;
    std::map<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy> layersWithRollbackStrategy;
    // Layers which got same rollback strategy twice
    std::set<mlir::Operation*> layersWithConvergedStrategy;
    // Keeps a record of operations where spill cost has already being taken into account
    std::set<mlir::Operation*> opsWithSpillWriteCounted;
    mlir::FuncOp _func;
    Logger _log;
    LayerCostModel _layerCostModel;
};

}  // namespace VPU
}  // namespace vpux
