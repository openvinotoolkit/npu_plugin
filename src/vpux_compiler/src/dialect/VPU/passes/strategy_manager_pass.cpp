//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/mc_strategy_getter_factory.hpp"
#include "vpux/compiler/dialect/VPU/mc_strategy_getter_interface.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPU/operation_strategies.hpp"

namespace vpux::VPU {
namespace {

//
// StrategyManagerImplPass
//

class StrategyManagerImplPass final : public StrategyManagerImplBase<StrategyManagerImplPass> {
public:
    explicit StrategyManagerImplPass(bool enablePrefetchTiling, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    SmallVector<std::pair<Strategy, StrategyCost>> getOperationOptions(mlir::Operation* operation);
    SmallVector<VPU::MultiClusterStrategy> getAvailiableStrategies(ArchKind arch, int64_t numClusters) const;

    std::shared_ptr<LayerVPUNNCost> _costModel;
    SmallVector<VPU::MultiClusterStrategy> _archStrategies;
    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult StrategyManagerImplPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading enablePrefetchTiling with an MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

SmallVector<std::pair<Strategy, StrategyCost>> StrategyManagerImplPass::getOperationOptions(
        mlir::Operation* operation) {
    SmallVector<std::pair<Strategy, StrategyCost>> strategies;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);

    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation);

    if (clusteredOp == nullptr && tilingBuilderOp == nullptr) {
        return strategies;
    }

    for (auto strategy : _archStrategies) {
        if (clusteredOp != nullptr &&
            !(clusteredOp.checkStrategyCompatibility(strategy) &&
              isStrategyCompatibleShape(clusteredOp, getShape(clusteredOp->getResult(0)), strategy, _log))) {
            continue;
        }

        bool hasPrefetch = false;
        do {
            mlir::ArrayAttr tilingStrategy;
            OutputTiling operationTiling;

            if (clusteredOp != nullptr) {
                clusteredOp.setMultiClusterStrategy(strategy);
            }

            auto mode = TilingMode::ISOLATED;
            if (opNeedsTiling(operation, hasPrefetch, _log)) {
                const auto layerTilingResult = getLayerTilingStrategy(tilingBuilderOp, hasPrefetch, _log, mode);

                if (mlir::failed(layerTilingResult)) {
                    continue;
                }

                operationTiling = layerTilingResult.getValue();
                VPUX_THROW_WHEN(operationTiling.empty(), "Couldn't get valid tiling for operation {0} in mode {1}",
                                operation->getLoc(), getTilingModeStr(mode));

                tilingStrategy = getIntArrayAttr(operation->getContext(), operationTiling[0].axis);
            } else if (clusteredOp != nullptr && !doesLayerFitIntoCMX(clusteredOp, strategy, Byte(0), _log)) {
                _log.trace("Layer {0} doesn't fit in CMX with strategy {1}", operation->getLoc(), strategy);
                break;
            }
            if (operation->hasAttr(multiClusterStrategy)) {
                operation->removeAttr(multiClusterStrategy);
            }

            const auto cost =
                    _costModel->getStrategyCost(operation, VPUNNCostParameters(strategy, operationTiling, hasPrefetch));

            _log.trace("For operation {0} with MC strategy {1}-{2} vpunn returns cost {3}", operation->getLoc(),
                       strategy, tilingStrategy, cost);

            strategies.emplace_back(Strategy(strategy, tilingStrategy, mode), cost);

            // in case prefetch tiling is enabled, both cases should be taken into account
            // with prefetching and without
            hasPrefetch ^= _enablePrefetchTiling;
        } while (hasPrefetch);
    }

    return strategies;
}

SmallVector<VPU::MultiClusterStrategy> StrategyManagerImplPass::getAvailiableStrategies(ArchKind arch,
                                                                                        int64_t numClusters) const {
    auto mcListGetter = createMCStrategyGetter(arch, numClusters);

    SmallVector<VPU::MultiClusterStrategy> strategies;
    mcListGetter->getMCStrategies(strategies);
    return strategies;
}

void StrategyManagerImplPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    _costModel = std::make_shared<LayerVPUNNCost>(func);
    _archStrategies = getAvailiableStrategies(VPU::getArch(module),
                                              IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE).count());

    // calculate cost for all possible strategies
    // assign strategy with min cost
    OperationStrategies operationStrategies;
    const auto findStrategyCallback = [&](mlir::Operation* operation) {
        auto strategies = getOperationOptions(operation);

        if (strategies.empty()) {
            return;
        }

        for (auto& [strategy, cost] : strategies) {
            auto operationStrategy = std::make_pair(operation, strategy);
            if (operationStrategies.hasStrategy(operationStrategy)) {
                operationStrategies.setStrategy(operationStrategy,
                                                std::min(cost, operationStrategies.getStrategyCost(operationStrategy)));
            } else {
                operationStrategies.addStrategy(operationStrategy, cost);
            }
        }

        // set current and best one
        auto minCostIt = std::min_element(strategies.begin(), strategies.end(), [](auto lhs, auto rhs) {
            return lhs.second < rhs.second;
        });

        if (minCostIt == strategies.end()) {
            return;
        }

        const auto minCostStrategy = std::make_pair(operation, minCostIt->first);
        operationStrategies.setCurrentStrategy(minCostStrategy);
        operationStrategies.setBestStrategy(minCostStrategy);
    };

    func->walk(findStrategyCallback);

    // TODO strategy optimization

    // set best strategy to each operation
    const auto setStrategyCallback = [&](mlir::Operation* operation) {
        if (VPU::isPureViewOp(operation) || !operationStrategies.hasAnyStrategy(operation)) {
            return;
        }

        const auto bestResult = operationStrategies.getBestStrategy(operation);

        if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
            clusteredOp.setMultiClusterStrategy(bestResult.getMCStrategy());
        }

        if (bestResult.getTilingStrategy() != nullptr) {
            if (auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation)) {
                tilingOp->setAttr(tilingStrategy, bestResult.getTilingStrategy());
            }
        }
    };

    func->walk(setStrategyCallback);
}

}  // namespace

//
// createStrategyManagerImplPass
//

std::unique_ptr<mlir::Pass> createStrategyManagerImplPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<StrategyManagerImplPass>(enablePrefetchTiling, log);
}

}  // namespace vpux::VPU
