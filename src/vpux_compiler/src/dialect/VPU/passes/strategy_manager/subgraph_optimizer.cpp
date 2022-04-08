//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/VPU/subgraph_optimizer.hpp"
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

SubgraphOptimizer::SubgraphOptimizer(mlir::FuncOp func, Logger log)
        : _func(func), _log(log), _layerCostModel(LayerCostModel(func, log)) {
}

VPU::MultiClusterStrategy SubgraphOptimizer::getRollbackStrategy(VPU::ClusteredOpInterface clusteredOp) {
    auto it = layersWithRollbackStrategy.find(clusteredOp);
    if (it != layersWithRollbackStrategy.end()) {
        return it->second;
    }

    return _layerCostModel.getMultiClusterStrategyValue(clusteredOp);
}

bool SubgraphOptimizer::isStrategySOKLike(VPU::ClusteredOpInterface op) {
    auto strategy = getRollbackStrategy(op);

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }
    return false;
}

bool SubgraphOptimizer::isStrategySOHLike(VPU::ClusteredOpInterface op) {
    auto strategy = getRollbackStrategy(op);

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return true;
    }
    return false;
}

bool SubgraphOptimizer::isValidStrategy(VPU::ClusteredOpInterface nceOp, VPU::MultiClusterStrategy strategy) {
    auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(nceOp.getOperation());
    if (!clusteredOp.checkStrategyCompatibility(strategy)) {
        return false;
    }

    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(nceOp->getName());
    auto isCompatibleStrategy = [&](VPU::ClusteredOpInterface op, VPU::MultiClusterStrategy targetStrategy) {
        const auto arch = VPU::getArch(nceOp);
        const auto isChannelMajor = (DimsOrder::fromValue(nceOp->getOperand(0)) == DimsOrder::NCHW) &&
                                    VPU::NCEInvariant::isChannelMajorCompatible(
                                            arch, nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>());
        const auto isCompressConv = VPU::NCEInvariant::isCompressConvolution(arch, nceOp);
        auto isCompatible = false;
        switch (targetStrategy) {
        case MultiClusterStrategy::SplitOverHeightOverlapped:
            isCompatible = (isChannelMajor || isCompressConv) &&
                           layerStrategyChecker->isOperationSplitOverHeightCompatible(op);
            break;
        case MultiClusterStrategy::SplitOverHeight:
            isCompatible = !isChannelMajor && !isCompressConv &&
                           layerStrategyChecker->isOperationSplitOverHeightCompatible(op);
            break;
        case MultiClusterStrategy::SplitOverKernel:
            isCompatible = layerStrategyChecker->isOperationSplitOverKernelCompatible(op);
            break;
        case MultiClusterStrategy::HKSwitch:
            isCompatible = layerStrategyChecker->isOperationSplitOverHeightCompatible(op);
            break;
        case MultiClusterStrategy::Clustering:
            isCompatible = true;
            break;
        default:
            VPUX_THROW("Unsupported strategy {0} for check nce op compatibility", targetStrategy);
        }
        return isCompatible;
    };

    return isCompatibleStrategy(nceOp, strategy);
}

/// @brief Return SOK-like strategie with least cost
/// @details SOK-like strategy may still have input/output spilling in some cases,
/// So we need to consider spilling costs when rollback
VPU::MultiClusterStrategy SubgraphOptimizer::getBestInSOKLikeStrategies(VPU::ClusteredOpInterface clusteredOp) {
    double HKCost = _layerCostModel.COST_MAX;
    auto nceOp = mlir::dyn_cast<NCEOpInterface>(clusteredOp.getOperation());

    if (isValidStrategy(clusteredOp, VPU::MultiClusterStrategy::HKSwitch)) {
        HKCost = _layerCostModel.getDPUandDMATimeCost(nceOp, VPU::MultiClusterStrategy::HKSwitch);
        _log.trace("HKSwitch has compute cost {0}", HKCost);
        auto spillingCost =
                getInputSpillingRollbackCostToMultiClusterLayer(clusteredOp, VPU::MultiClusterStrategy::HKSwitch) +
                getOutputSpillingRollbackCostToMultiClusterLayer(clusteredOp, VPU::MultiClusterStrategy::HKSwitch);
        HKCost += spillingCost;
        _log.trace("HKSwitch has spilling cost {0}", spillingCost);
    }

    double SOKCost = _layerCostModel.COST_MAX;
    if (isValidStrategy(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel)) {
        SOKCost = _layerCostModel.getDPUandDMATimeCost(nceOp, VPU::MultiClusterStrategy::SplitOverKernel);
        _log.trace("SplitOverKernel has compute cost {0}", SOKCost);
        auto spillingCost = getInputSpillingRollbackCostToMultiClusterLayer(
                                    clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel) +
                            getOutputSpillingRollbackCostToMultiClusterLayer(
                                    clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel);
        SOKCost += spillingCost;
        _log.trace("SplitOverKernel has spilling cost {0}", spillingCost);
    }

    double clusteringCost = _layerCostModel.COST_MAX;
    if (isValidStrategy(clusteredOp, VPU::MultiClusterStrategy::Clustering)) {
        clusteringCost = _layerCostModel.getDPUandDMATimeCost(nceOp, VPU::MultiClusterStrategy::Clustering);
        _log.trace("Clustering has compute cost {0}", clusteringCost);
        auto spillingCost =
                getInputSpillingRollbackCostToMultiClusterLayer(clusteredOp, VPU::MultiClusterStrategy::Clustering) +
                getOutputSpillingRollbackCostToMultiClusterLayer(clusteredOp, VPU::MultiClusterStrategy::Clustering);
        clusteringCost += spillingCost;
        _log.trace("Clustering has spilling cost {0}", spillingCost);
    }

    if ((HKCost < SOKCost) && (HKCost < clusteringCost)) {
        _log.trace("HKSwitch is selected with cost {0}", HKCost);
        return VPU::MultiClusterStrategy::HKSwitch;
    }

    // Sometimes Clustering strategy can avoid spilling, which makes it has less overall cost than SOK
    // But spilling could be overlapped with DPU Task during scheduling, then actually SOK gives better performance.
    // Meanwhile inaccurate cost calcution may mislead the strategy selection. We can compare the cost between SOK and
    // Clustering again after intergrating VPUNN.
    if (SOKCost != _layerCostModel.COST_MAX) {
        _log.trace("SplitOverKernel is selected with cost {0}", SOKCost);
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }
    if (clusteringCost != _layerCostModel.COST_MAX) {
        _log.trace("Clustering is selected with cost {0}", clusteringCost);
        return VPU::MultiClusterStrategy::Clustering;
    }

    // Here means no SOK-like strategy available
    // return original strategy as result
    return _layerCostModel.getMultiClusterStrategyValue(clusteredOp);
}

/// @brief Return SOH-like strategie with least cost
/// @details SOH-like strategy may still have input/output spilling in some cases,
/// So we need to consider spilling costs when rollback
/// Currently we only consider SplitOverHeight strategy, but we still calculate the cost for debugging purpose
VPU::MultiClusterStrategy SubgraphOptimizer::getBestInSOHLikeStrategies(VPU::ClusteredOpInterface clusteredOp) {
    double SOHCost = _layerCostModel.COST_MAX;
    auto nceOp = mlir::dyn_cast<NCEOpInterface>(clusteredOp.getOperation());
    if (isValidStrategy(clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight)) {
        SOHCost = _layerCostModel.getDPUandDMATimeCost(nceOp, VPU::MultiClusterStrategy::SplitOverHeight);
        _log.trace("SplitOverHeight has compute cost {0}", SOHCost);
        auto spillingCost = getInputSpillingRollbackCostToMultiClusterLayer(
                                    clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight) +
                            getOutputSpillingRollbackCostToMultiClusterLayer(
                                    clusteredOp, VPU::MultiClusterStrategy::SplitOverHeight);
        SOHCost += spillingCost;
        _log.trace("SplitOverHeight has spilling cost {0}", spillingCost);
    }

    if (SOHCost != _layerCostModel.COST_MAX) {
        _log.trace("SplitOverHeight is selected with cost {0}", SOHCost);
        return VPU::MultiClusterStrategy::SplitOverHeight;
    }

    // Here means no SOH-like strategy available
    // return original strategy as result
    return _layerCostModel.getMultiClusterStrategyValue(clusteredOp);
}

/// @brief Get the input spilling cost with rollback strategies (cycles)
/// @details When we calculate spilling cost for current layer with rollback strategy, we need to make sure the
/// neighboring layer is using rollback strategy as well if it has one
double SubgraphOptimizer::getInputSpillingRollbackCostToMultiClusterLayer(VPU::ClusteredOpInterface nceOp,
                                                                          VPU::MultiClusterStrategy rollbackStrategy) {
    return llvm::TypeSwitch<mlir::Operation*, double>(nceOp.getOperation())
            .Case<NCEMaxPoolOp>([&](NCEMaxPoolOp origOp) {
                return getInputSpillingRollbackCostToMultiClusterLayer(origOp, origOp.input(), rollbackStrategy);
            })
            .Case<NCEAveragePoolOp>([&](NCEAveragePoolOp origOp) {
                return getInputSpillingRollbackCostToMultiClusterLayer(origOp, origOp.input(), rollbackStrategy);
            })
            .Case<NCEEltwiseOp>([&](NCEEltwiseOp origOp) {
                return getInputSpillingRollbackCostToMultiClusterLayer(origOp, origOp.input1(), rollbackStrategy) +
                       getInputSpillingRollbackCostToMultiClusterLayer(origOp, origOp.input2(), rollbackStrategy);
            })
            .Case<NCEConvolutionOp>([&](NCEConvolutionOp origOp) {
                return getInputSpillingRollbackCostToMultiClusterLayer(origOp, origOp.input(), rollbackStrategy);
            })
            .Case<NCEDepthConvolutionOp>([&](NCEDepthConvolutionOp origOp) {
                return getInputSpillingRollbackCostToMultiClusterLayer(origOp, origOp.input(), rollbackStrategy);
            })
            .Default([&](mlir::Operation*) {
                VPUX_THROW("Find rollback strategy {0} for non-NCE Task {1}", rollbackStrategy, nceOp->getLoc());
                return 0.0;
            });
}

/// @brief The input spilling cost for specified input operand
double SubgraphOptimizer::getInputSpillingRollbackCostToMultiClusterLayer(VPU::ClusteredOpInterface clusterOp,
                                                                          mlir::Value input,
                                                                          VPU::MultiClusterStrategy rollbackStrategy) {
    auto targetTensorType = _layerCostModel.getDistributedInputType(clusterOp, input.getDefiningOp(), rollbackStrategy);

    auto parent = input.getDefiningOp();
    if (parent == nullptr) {
        return _layerCostModel.getSpillingReadCost(targetTensorType);
    }

    if (mlir::isa<VPU::ShapeCastOp>(parent)) {
        // propagate ShapeCast
        parent = parent->getOperand(0).getDefiningOp();
        if (parent == nullptr) {
            return _layerCostModel.getSpillingReadCost(targetTensorType);
        }
    }

    return llvm::TypeSwitch<mlir::Operation*, double>(parent)
            .Case<VPU::ClusteredOpInterface>([&](VPU::ClusteredOpInterface parentOp) {
                if (!_layerCostModel.hasMultiClusterStrategy(parentOp))
                    return 0.0;

                VPU::MultiClusterStrategy parentStrategy = getRollbackStrategy(parentOp);
                auto currentSpillingCost =
                        _layerCostModel.calculateSpillingCost(parentOp, clusterOp, parentStrategy, rollbackStrategy);
                return currentSpillingCost.writeCost + currentSpillingCost.readCost;
            })

            .Case<IE::ConcatOp>([&](IE::ConcatOp parentOp) {
                // NOTE: If the concat is a CMX concat and the outputs are duplicated in each cluster, there is
                // supposed to be no spilling when strategy is Clustering or SplitOverKernel. For now it's just
                // a workaround to check the strategy only since we don't know whether it's a cmx concat or not now.
                bool needSpilling = true;
                if (rollbackStrategy == MultiClusterStrategy::SplitOverKernel ||
                    rollbackStrategy == MultiClusterStrategy::Clustering) {
                    SmallVector<VPU::ClusteredOpInterface> nceInputOps;
                    for (auto concatInput : parentOp.inputs()) {
                        if (auto nceInput = concatInput.getDefiningOp<VPU::ClusteredOpInterface>()) {
                            nceInputOps.push_back(nceInput);
                        }
                    }
                    auto hasOutputSpilling = [&](VPU::ClusteredOpInterface nceInput) {
                        if (!_layerCostModel.hasMultiClusterStrategy(nceInput)) {
                            return true;
                        }
                        auto nceInputStrategy = getRollbackStrategy(nceInput);
                        auto requireTiling = _layerCostModel.doesLayerRequireTiling(nceInput, nceInputStrategy) ||
                                             _layerCostModel.doesLayerRequireTiling(clusterOp, rollbackStrategy);
                        return requireTiling ||
                               _layerCostModel.hasSpilling(nceInput, nceInputStrategy, clusterOp, rollbackStrategy);
                    };
                    needSpilling = nceInputOps.empty() || llvm::any_of(nceInputOps, hasOutputSpilling);
                }
                return needSpilling ? _layerCostModel.getSpillingReadCost(targetTensorType) : 0.0;
            })
            .Default([&](mlir::Operation*) {
                return _layerCostModel.getSpillingReadCost(targetTensorType);
            });
}

/// @brief Get the output spilling cost with rollback strategies (cycles)
/// @details When we calculate spilling cost for current layer with rollback strategy, we need to make sure the
/// neighboring layer is using rollback strategy as well if it has one
double SubgraphOptimizer::getOutputSpillingRollbackCostToMultiClusterLayer(VPU::ClusteredOpInterface nceOp,
                                                                           VPU::MultiClusterStrategy rollbackStrategy) {
    bool hasCalculatedSpillingWriteCost = false;
    double totalSpillingCost = 0.0;
    for (auto user : nceOp->getResult(0).getUsers()) {
        if (mlir::isa<VPU::QuantizeCastOp>(user) || (mlir::isa<VPU::ShapeCastOp>(user) && user->hasOneUse())) {
            // propagate cast ops
            user = *user->getResult(0).getUsers().begin();
        }
        if (user == nullptr || !_layerCostModel.hasMultiClusterStrategy(user)) {
            continue;
        }

        if (auto userOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(user)) {
            auto userStrategy = getRollbackStrategy(userOp);
            auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(nceOp.getOperation());
            auto spillingCost =
                    _layerCostModel.calculateSpillingCost(clusteredOp, userOp, rollbackStrategy, userStrategy);
            if (!hasCalculatedSpillingWriteCost) {
                totalSpillingCost += spillingCost.writeCost;
                hasCalculatedSpillingWriteCost = true;
            }
            totalSpillingCost += spillingCost.readCost;
        }
    }

    return totalSpillingCost;
}

/// @brief The idea of this algorithm is to avoid spilling by optimizing strategy
/// @details We execute following steps for each NCE task from input to output through a topological sort of the Op
/// Model.
/// 1. Check if a source NCE Task needs rollback strategy
/// 2. Find the best rollback strategy for the source task
/// 3. Check neighboring NCE Task. Add to queue if it needs rollback strategy.
/// 4. Execute step 2-3 for each task in queue until the queue is empty
/// @example List some typical scenarios
/// 1. {SOH -> SOH -> SOK} convert to {SOH -> HK -> SOK}
/// 2. {SOK -> SOK -> SOH} convert to {SOH -> SOH -> SOH}
/// 3. {SOH -> SOK -> SOH} convert to {SOH -> SOH -> SOH}
/// 4. {SOK -> SOH -> SOK} convert to {SOH -> HK -> SOK}
/// @todo This algorithm is a local strategy optimization for subgraphs. The final strategy maybe not global optimized.
/// For example, {SOK -> SOK -> SOH} maybe better converting to {SOK -> SOK -> SOK}
void SubgraphOptimizer::optimizeStrategyAvoidSpillingOnSubgraph(VPU::ClusteredOpInterface clusteredOp) {
    if (!_layerCostModel.hasMultiClusterStrategy(clusteredOp) ||
        (!_layerCostModel.hasOutputSpillingToMultiClusterLayer(clusteredOp) &&
         !_layerCostModel.hasInputSpillingToMultiClusterLayer(clusteredOp))) {
        return;
    }

    // A layer will have spilling regardless of multi-cluster strategy if current layer needs be tiled because CMX
    // memory is not enough. The situation will change when we enable vertical fusion. There is no benifit for
    // performance to do subgraph optimization in such case because spilling can't be removed anyways.
    if (_layerCostModel.doesLayerRequireTiling(clusteredOp,
                                               _layerCostModel.getMultiClusterStrategyValue(clusteredOp))) {
        return;
    }

    _log.trace("Subgraph opt: beginning node '{0}'", clusteredOp->getLoc());

    // If the source NCE task has SOK like strategy, we consider SOH rollback strategy because SOK usually has spilling
    // to SOH. And SOH - SOH can avoid spilling.
    // If the source NCE task has SOH like strategy, we consider SOK/Clustering/HKSwitch rollback strategy because it's
    // more possible to avoid spilling.
    bool rollbackToSOH = isStrategySOKLike(clusteredOp);
    layersNeedRollback.clear();
    layersWithRollbackStrategy.clear();
    layersWithConvergedStrategy.clear();
    opsWithSpillWriteCounted.clear();

    // Processing SOH -> SOK/Clustering
    VPU::ClusteredOpInterface currentOp = clusteredOp;
    VPU::MultiClusterStrategy rollbackStrategy;
    layersNeedRollback.push_back(currentOp);
    while (!layersNeedRollback.empty()) {
        currentOp = layersNeedRollback.front();
        auto clusteredCurrentOp = mlir::dyn_cast<ClusteredOpInterface>(currentOp.getOperation());
        layersNeedRollback.erase(layersNeedRollback.begin());
        rollbackStrategy =
                rollbackToSOH ? getBestInSOHLikeStrategies(currentOp) : getBestInSOKLikeStrategies(currentOp);
        // Sometimes a layer triggers rollback several times in a subgraph because we allows rollback strategy be
        // changed during optimization. It makes sense when we find rollback strategy A first but later a better
        // rollback strategy B is found. However if B is equal to A, it means the rollback strategy is converged. We
        // need to stop the iteration in such case otherwise it could be infinite loop.
        if ((layersWithRollbackStrategy.find(currentOp) != layersWithRollbackStrategy.end()) &&
            (rollbackStrategy == layersWithRollbackStrategy[currentOp])) {
            _log.trace("  Layer '{0} has been processed twice with same strategy {1}", currentOp->getLoc(),
                       rollbackStrategy);
            layersWithConvergedStrategy.insert(currentOp);
        } else {
            layersWithRollbackStrategy[currentOp] = rollbackStrategy;
            _log.trace("  Layer '{0} has been processed, candidate strategy {1}", currentOp->getLoc(),
                       rollbackStrategy);
        }

        // A layer doesn't need rollback strategy, if
        // 1. it has no multi-cluster strategy
        // 2. it has SOK like strategy and we want SOK like rollback
        // 3. it has SOH like strategy and we want SOH like rollback
        // 4. it requires tiling
        // 5. is SW Layers, strategies are fixed
        // TODO extend to support SW layers E#63362
        const auto doesLayerNeedRollbackStrategy = [&](mlir::Operation* targetOp) -> bool {
            if (!_layerCostModel.hasMultiClusterStrategy(targetOp)) {
                _log.trace("Find edge node {0} without strategy", targetOp->getLoc());
                return false;
            }
            if (auto swOp = mlir::dyn_cast<SWOpInterface>(targetOp)) {
                _log.trace("SW not supported {0}", swOp->getLoc());
                return false;
            }
            auto clusteredTargetOp = mlir::dyn_cast<ClusteredOpInterface>(targetOp);

            if (isStrategySOKLike(targetOp) && !rollbackToSOH) {
                _log.trace("Find edge node {0} with strategy {1}", targetOp->getLoc(),
                           getRollbackStrategy(clusteredTargetOp));
                return false;
            }

            if (isStrategySOHLike(targetOp) && rollbackToSOH) {
                _log.trace("Find edge node {0} with strategy {1}", targetOp->getLoc(),
                           getRollbackStrategy(clusteredTargetOp));
                return false;
            }

            if (_layerCostModel.doesLayerRequireTiling(targetOp, getRollbackStrategy(clusteredTargetOp))) {
                _log.trace("Find edge node {0} requires tiling with strategy {1}", targetOp->getLoc(),
                           getRollbackStrategy(clusteredTargetOp));
                return false;
            }

            return true;
        };

        // Check if the child layer needs rollback strategy
        for (auto child : currentOp->getResult(0).getUsers()) {
            if ((std::find(layersNeedRollback.begin(), layersNeedRollback.end(), child) != layersNeedRollback.end()) ||
                (layersWithConvergedStrategy.find(child) != layersWithConvergedStrategy.end())) {
                continue;
            }

            if (doesLayerNeedRollbackStrategy(child) &&
                _layerCostModel.hasSpilling(clusteredCurrentOp, rollbackStrategy, child)) {
                layersNeedRollback.push_back(child);
                _log.trace("    Push child '{0} to layersNeedRollback queue", child->getLoc());
            }
        }

        // Check if the parent layer needs rollback strategy
        SmallVector<mlir::Value> layerInputs;
        if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(currentOp.getOperation())) {
            layerInputs.push_back(eltwiseOp.input1());
            layerInputs.push_back(eltwiseOp.input2());
        } else {
            layerInputs.push_back(currentOp->getOperand(0));
        }

        for (auto input : layerInputs) {
            auto parent = input.getDefiningOp();
            if ((parent == nullptr) ||
                (std::find(layersNeedRollback.begin(), layersNeedRollback.end(), parent) != layersNeedRollback.end()) ||
                (layersWithConvergedStrategy.find(parent) != layersWithConvergedStrategy.end())) {
                continue;
            }
            auto clusteredParentOp = mlir::dyn_cast<ClusteredOpInterface>(parent);
            if (doesLayerNeedRollbackStrategy(parent) &&
                _layerCostModel.hasSpilling(clusteredParentOp, clusteredCurrentOp, rollbackStrategy)) {
                layersNeedRollback.push_back(parent);
                _log.trace("    Push parent '{0} to layersNeedRollback queue", parent->getLoc());
            }
        }
    }

    // Calculate original cost and rollback cost
    double originalCost = 0;
    double rollbackCost = 0;
    std::set<mlir::Operation*> opsWithOutputSpillingCounted;
    for (auto opWithStrategy : layersWithRollbackStrategy) {
        auto clusteredTask = opWithStrategy.first;
        auto newStrategy = opWithStrategy.second;
        auto nceTask = mlir::dyn_cast<NCEOpInterface>(clusteredTask.getOperation());
        auto oldStrategy = _layerCostModel.getMultiClusterStrategyValue(clusteredTask);

        // compute cost + weights dma cost
        auto originalBasicCost = _layerCostModel.getDPUandDMATimeCost(nceTask, oldStrategy);
        auto rollbackBasicCost = _layerCostModel.getDPUandDMATimeCost(nceTask, newStrategy);
        _log.trace("add originalCost cost {0} for op {1} with strategy {2}", originalBasicCost, clusteredTask->getLoc(),
                   oldStrategy);
        _log.trace("add rollback cost {0} for op {1} with strategy {2}", rollbackBasicCost, clusteredTask->getLoc(),
                   newStrategy);

        originalCost += originalBasicCost;
        rollbackCost += rollbackBasicCost;

        // input spilling cost is calculated by the output spilling cost of parent
        SmallVector<mlir::Value> layerInputs;
        if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(clusteredTask.getOperation())) {
            layerInputs.push_back(eltwiseOp.input1());
            layerInputs.push_back(eltwiseOp.input2());
        } else {
            layerInputs.push_back(clusteredTask->getOperand(0));
        }

        for (auto input : layerInputs) {
            auto parent = input.getDefiningOp();
            if ((parent != nullptr) && (_layerCostModel.hasMultiClusterStrategy(parent)) &&
                (!opsWithOutputSpillingCounted.count(parent))) {
                auto parentOldStrategy = _layerCostModel.getMultiClusterStrategyValue(parent);
                auto clusteredParent = mlir::dyn_cast<ClusteredOpInterface>(parent);
                auto parentNewStrategy = getRollbackStrategy(clusteredParent);

                auto originalInputSpillingCost =
                        _layerCostModel.getOutputSpillingCostToMultiClusterLayer(parent, parentOldStrategy);
                auto rollbackInputSpillingCost =
                        getOutputSpillingRollbackCostToMultiClusterLayer(parent, parentNewStrategy);
                _log.trace("add originalCost input Spilling {0} for op {1} with strategy {2}",
                           originalInputSpillingCost, parent->getLoc(), parentOldStrategy);
                _log.trace("add rollback input Spilling {0} for op {1} with strategy {2}", rollbackInputSpillingCost,
                           parent->getLoc(), parentNewStrategy);

                originalCost += originalInputSpillingCost;
                rollbackCost += rollbackInputSpillingCost;
                opsWithOutputSpillingCounted.insert(parent);
            }
        }

        // output spilling cost
        if (!opsWithOutputSpillingCounted.count(clusteredTask)) {
            auto originalOutputSpillingCost =
                    _layerCostModel.getOutputSpillingCostToMultiClusterLayer(clusteredTask, oldStrategy);
            auto rollbackOutputSpillingCost =
                    getOutputSpillingRollbackCostToMultiClusterLayer(clusteredTask, newStrategy);
            _log.trace("add originalCost output Spilling {0} for op {1} with strategy {2}", originalOutputSpillingCost,
                       clusteredTask->getLoc(), oldStrategy);
            _log.trace("add rollback output Spilling {0} for op {1} with strategy {2}", rollbackOutputSpillingCost,
                       clusteredTask->getLoc(), newStrategy);

            originalCost += originalOutputSpillingCost;
            rollbackCost += rollbackOutputSpillingCost;
            opsWithOutputSpillingCounted.insert(clusteredTask);
        }
    }

    if (rollbackCost < originalCost) {
        for (auto& opWithStrategy : layersWithRollbackStrategy) {
            auto clusteredTask = opWithStrategy.first;
            auto newStrategy = opWithStrategy.second;
            clusteredTask.setMultiClusterStrategyAttr(newStrategy);
            _log.trace("  [rollback] '{0}' : set strategy as {1}", clusteredTask->getLoc(), newStrategy);
        }
    } else {
        _log.trace("Subgraph opt: rollback unneccessary! Strategies no change");
    }
    _log.trace("  Rollback cost: {0} , Current cost: {1}", rollbackCost, originalCost);
}

void SubgraphOptimizer::optimizeStrategyAvoidSpillingOnModel() {
    const auto callback = [this](VPU::ClusteredOpInterface clusteredOp) {
        // skip SW Layers, strategies are fixed
        // TODO extend to support SW layers E#63362
        if (mlir::isa<SWOpInterface>(clusteredOp.getOperation())) {
            return;
        }
        optimizeStrategyAvoidSpillingOnSubgraph(clusteredOp);
    };

    /// @brief Traversing nodes with preOrder to execute subgraph optimization
    _func.walk(callback);
}
