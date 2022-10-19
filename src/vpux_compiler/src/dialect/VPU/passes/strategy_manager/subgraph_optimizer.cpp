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

#include "vpux/compiler/dialect/VPU/subgraph_optimizer.hpp"
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

SubgraphOptimizer::SubgraphOptimizer(mlir::FuncOp func, Logger log)
        : _func(func), _log(log), _layerCostModel(LayerCostModel(func, log)) {
}

/// @brief Strategy is SOK-like means the output tensor mode has DUPLICATED feature,
/// and allowHK as an extra switcher for HKSwitch
bool SubgraphOptimizer::isStrategySOKLike(VPU::NCEOpInterface op, bool allowHK) {
    auto strategyAttr = op->getAttr(multiClusterStrategy);
    if (!strategyAttr) {
        VPUX_THROW("isStrategySOKLike member func expects NCE op with multiClusterStrategy attribute");
    }
    const auto strategy = strategyAttr.cast<VPU::MultiClusterStrategyAttr>().getValue();
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering ||
        (allowHK && strategy == VPU::MultiClusterStrategy::HKSwitch)) {
        return true;
    }
    return false;
}

bool SubgraphOptimizer::isValidStrategy(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) {
    if (!nceOp.checkStrategyCompatibility(strategy)) {
        return false;
    }

    auto layerStrategyChecker = LayerStrategyCheckerFactory::instance().get(nceOp->getName());
    auto isCompatibleStrategy = [&](VPU::NCEOpInterface op, VPU::MultiClusterStrategy targetStrategy) {
        const auto arch = VPU::getArch(nceOp);
        const auto isChannelMajor = (DimsOrder::fromValue(nceOp->getOperand(0)) == DimsOrder::NCHW) &&
                                    VPU::NCEInvariant::isChannelMajorCompatible(
                                            arch, nceOp->getOperand(0).getType().cast<vpux::NDTypeInterface>());
        auto isCompatible = false;
        switch (targetStrategy) {
        case MultiClusterStrategy::SplitOverHeightOverlapped:
            isCompatible = isChannelMajor && layerStrategyChecker->isOperationSplitOverHeightCompatible(op);
            break;
        case MultiClusterStrategy::SplitOverHeight:
            isCompatible = !isChannelMajor && layerStrategyChecker->isOperationSplitOverHeightCompatible(op);
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
    // We should only check that a layer fits into CMX after we have confirmed
    // that the layer has been assigned a valid multi-cluster strategy because the strategy determines the amount of CMX
    // required
    return isCompatibleStrategy(nceOp, strategy);
}

VPU::NCEOpInterface SubgraphOptimizer::findSourceOperation(VPU::NCEOpInterface currentOp, bool isParent) {
    if (isParent) {
        // find the source child op
        for (auto child : currentOp->getResult(0).getUsers()) {
            if (processedOps.find(child) != processedOps.end()) {
                return child;
            }
        }
    } else {
        // find the source parent op
        for (auto input : currentOp->getOperands()) {
            auto parent = input.getDefiningOp();
            if (processedOps.find(parent) != processedOps.end()) {
                return parent;
            }
        }
    }
    return nullptr;
}

LayerCostModel::SpillingCost SubgraphOptimizer::getSpillingCostWithSourceOp(VPU::NCEOpInterface currentOp,
                                                                            VPU::NCEOpInterface sourceOp,
                                                                            VPU::MultiClusterStrategy currentStrategy,
                                                                            bool isParent) {
    auto currentStrategyAttr = VPU::MultiClusterStrategyAttr::get(currentOp->getContext(), currentStrategy);
    if (isParent) {
        // Add output spilling with source op (child)
        if (sourceOp != nullptr) {
            if (parentsToChange.find(sourceOp) != parentsToChange.end()) {
                auto sourceOpMC = parentsToChange[sourceOp];
                auto sourceOpMCAttr = VPU::MultiClusterStrategyAttr::get(sourceOp->getContext(), sourceOpMC);
                auto spillingCost =
                        _layerCostModel.calculateSpillingCost(currentOp, sourceOp, currentStrategyAttr, sourceOpMCAttr);
                return spillingCost;
            } else if (childrenToChange.find(sourceOp) != childrenToChange.end()) {
                auto sourceOpMC = childrenToChange[sourceOp];
                auto sourceOpMCAttr = VPU::MultiClusterStrategyAttr::get(sourceOp->getContext(), sourceOpMC);
                auto spillingCost =
                        _layerCostModel.calculateSpillingCost(currentOp, sourceOp, currentStrategyAttr, sourceOpMCAttr);
                return spillingCost;
            }
        }
    } else {
        // Add input spilling with source op (parent)
        if (sourceOp != nullptr) {
            if (parentsToChange.find(sourceOp) != parentsToChange.end()) {
                auto sourceOpMC = parentsToChange[sourceOp];
                auto sourceOpMCAttr = VPU::MultiClusterStrategyAttr::get(sourceOp->getContext(), sourceOpMC);
                auto spillingCost =
                        _layerCostModel.calculateSpillingCost(sourceOp, currentOp, sourceOpMCAttr, currentStrategyAttr);
                return spillingCost;
            } else if (childrenToChange.find(sourceOp) != childrenToChange.end()) {
                auto sourceOpMC = childrenToChange[sourceOp];
                auto sourceOpMCAttr = VPU::MultiClusterStrategyAttr::get(sourceOp->getContext(), sourceOpMC);
                auto spillingCost =
                        _layerCostModel.calculateSpillingCost(sourceOp, currentOp, sourceOpMCAttr, currentStrategyAttr);
                return spillingCost;
            }
        }
    }
    return {0.0, 0.0};
}

/// @brief Return {best_strategy, best_cost} in SOK-like strategies to accumulate to the total rollback cost
/// HKSwitch is the first option only if it's available
/// @details SOK-like strategy may still have input/output spilling in some special cases,
/// So we need to consider spilling costs when rollback
std::pair<VPU::MultiClusterStrategy, double> SubgraphOptimizer::getBestInSOKLikeStrategies(VPU::NCEOpInterface op,
                                                                                           bool isParent,
                                                                                           bool allowHK) {
    auto sourceOp = findSourceOperation(op, isParent);
    auto parentOp = isParent ? op : sourceOp;
    if (allowHK && isValidStrategy(op, VPU::MultiClusterStrategy::HKSwitch)) {
        double HKCost = _layerCostModel.getDPUandDMATimeCost(op, VPU::MultiClusterStrategy::HKSwitch);
        if (sourceOp != nullptr) {
            auto spillingCost =
                    getSpillingCostWithSourceOp(op, sourceOp, VPU::MultiClusterStrategy::HKSwitch, isParent);
            if (opsWithSpillWriteCounted.count(parentOp)) {
                HKCost += spillingCost.readCost;
            } else {
                HKCost += (spillingCost.readCost + spillingCost.writeCost);
                opsWithSpillWriteCounted.insert(parentOp);
            }
        }

        return {VPU::MultiClusterStrategy::HKSwitch, HKCost};
    }

    double SOKCost = _layerCostModel.COST_MAX;
    if (isValidStrategy(op, VPU::MultiClusterStrategy::SplitOverKernel)) {
        SOKCost = _layerCostModel.getDPUandDMATimeCost(op, VPU::MultiClusterStrategy::SplitOverKernel);
        if (sourceOp != nullptr) {
            auto spillingCost =
                    getSpillingCostWithSourceOp(op, sourceOp, VPU::MultiClusterStrategy::SplitOverKernel, isParent);
            if (opsWithSpillWriteCounted.count(parentOp)) {
                SOKCost += spillingCost.readCost;
            } else {
                SOKCost += (spillingCost.readCost + spillingCost.writeCost);
            }
        }
    }

    double clusteringCost = _layerCostModel.COST_MAX;
    if (isValidStrategy(op, VPU::MultiClusterStrategy::Clustering)) {
        clusteringCost = _layerCostModel.getDPUandDMATimeCost(op, VPU::MultiClusterStrategy::Clustering);
        if (sourceOp != nullptr) {
            auto spillingCost =
                    getSpillingCostWithSourceOp(op, sourceOp, VPU::MultiClusterStrategy::Clustering, isParent);
            if (opsWithSpillWriteCounted.count(parentOp)) {
                clusteringCost += spillingCost.readCost;
            } else {
                clusteringCost += (spillingCost.readCost + spillingCost.writeCost);
            }
        }
    }

    auto origMCStrategy = op->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();
    double origCost = _layerCostModel.getDPUandDMATimeCost(op, origMCStrategy);
    if (sourceOp != nullptr) {
        auto spillingCost = getSpillingCostWithSourceOp(op, sourceOp, origMCStrategy, isParent);
        if (opsWithSpillWriteCounted.count(parentOp)) {
            origCost += spillingCost.readCost;
        } else {
            origCost += (spillingCost.readCost + spillingCost.writeCost);
        }
    }

    // Record it to avoid spillWrite cost accumulated duplicately
    if (sourceOp != nullptr) {
        opsWithSpillWriteCounted.insert(parentOp);
    }

    if (SOKCost < clusteringCost) {
        return {VPU::MultiClusterStrategy::SplitOverKernel, SOKCost};
    }
    if (clusteringCost != _layerCostModel.COST_MAX) {
        return {VPU::MultiClusterStrategy::Clustering, clusteringCost};
    }

    // Here means no SOK-like strategy available
    // return original strategy as result
    return {origMCStrategy, origCost};
}

/// @brief The idea of this algorithm is to decide to rollback original strategy to avoid spilling,
/// rolling back SOH to some point where this strategy switch can happen in CMX (like HKSwitch)
/// We move backwards through a topological sort of the Op Model
/// @example Consider a simple linear graph A (SOH or HK) -> B (SOH) -> C (SOH) -> D (SOK)
/// When we reach C, a SOH layer that must spill we process it:
/// 1. Add C to ops_to_change, mark it
/// 2. Add unmarked children to Q_c, if they have SOH or HKSwitch strategy and mark them
///      ex: Q_c : remains empty
/// 3. If C cannnot take HKSwitch, add unmarked parents to Q_p and mark them
///      ex: Q_p : B
/// 4. Continue processing elements from Q_p while not empty
///      ex: pop B and process it from step 1 (Q_c will remain empty, Q_p: A)
///          pop A and process it from step 1 (Q_c will remain empty, Q_p is empty)
/// 5. Continue processing elements from Q_c while not empty
/// 6. If cost cheaper to roll back, last HK-eligible op added to ops to change is HK,
///    rest take best compatible (SOK, clustering) strategy
/// @todo Considering scenario like:
/// SOK -> SOH after greedy assignment, SOK can be optimized as SOH in some situations.
void SubgraphOptimizer::optimizeStrategyAvoidSpillingOnSubgraph(VPU::NCEOpInterface currentOp) {
    if (!currentOp->getAttr(multiClusterStrategy) || isStrategySOKLike(currentOp, true) ||
        !_layerCostModel.hasOutputSpilling(currentOp)) {
        return;
    }
    _log.trace("Subgraph opt: begining node '{0}'", currentOp->getLoc());

    double currentCost = 0.0;
    double rollbackCost = 0.0;
    parents = {};
    children = {};
    parentsToChange.clear();
    childrenToChange.clear();
    processedOps.clear();
    opsWithSpillWriteCounted.clear();

    // Processing SOH -> SOK/Clustering
    VPU::NCEOpInterface op = currentOp;
    std::pair<VPU::MultiClusterStrategy, double> strategyWithCost;
    parents.push(op);
    while (!parents.empty() || !children.empty()) {
        if (!parents.empty()) {
            op = parents.front();
            parents.pop();
            strategyWithCost = getBestInSOKLikeStrategies(op, true);
            rollbackCost += strategyWithCost.second;
            parentsToChange.insert({op, strategyWithCost.first});
            _log.trace("  Parent '{0} has been processed, candidate strategy {1}", op->getLoc(),
                       strategyWithCost.first);
        } else {
            op = children.front();
            children.pop();
            strategyWithCost = getBestInSOKLikeStrategies(op, false, false);
            rollbackCost += strategyWithCost.second;
            childrenToChange.insert({op, strategyWithCost.first});
            _log.trace("  Child '{0} has been processed, candidate strategy {1}", op->getLoc(), strategyWithCost.first);
        }

        auto greedyMCStrategy = op->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue();
        currentCost += _layerCostModel.getDPUandDMATimeCost(op, greedyMCStrategy);
        currentCost += _layerCostModel.getOutputSpillingCost(op, greedyMCStrategy);
        processedOps.insert(op);

        // Add invalid children to queue if they are SOH or HKSwitch
        for (auto child : op->getResult(0).getUsers()) {
            if (processedOps.find(child) != processedOps.end()) {
                continue;
            }
            if (child->getAttr(multiClusterStrategy)) {
                if (isStrategySOKLike(child, false)) {
                    // Process edge case for stop point: last output spilling
                    auto strategyAttr = VPU::MultiClusterStrategyAttr::get(op->getContext(), strategyWithCost.first);
                    auto lastSpillingCost = _layerCostModel.calculateSpillingCost(
                            op, child, strategyAttr,
                            child->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>());

                    if (opsWithSpillWriteCounted.count(op)) {
                        rollbackCost += lastSpillingCost.readCost;
                    } else {
                        rollbackCost += (lastSpillingCost.readCost + lastSpillingCost.writeCost);
                        opsWithSpillWriteCounted.insert(op);
                    }
                } else {
                    _log.trace("    Push child '{0} to children queue", child->getLoc());
                    children.push(child);
                }
            }
        }

        // We stop moving up the graph when we find nodes that could be HKSwitch
        // or nodes could be compatible with parent strategy.
        // Endpoint conditions:
        // 1. op can be HK: STOP
        // 2. op can be SOK and parent is SOK-like: STOP
        // 3. op can be Clustering and parent is SOK-like: STOP
        if (!isValidStrategy(op, VPU::MultiClusterStrategy::HKSwitch)) {
            for (auto input : op->getOperands()) {
                auto parent = input.getDefiningOp();
                if (parent == nullptr || processedOps.find(parent) != processedOps.end()) {
                    continue;
                }
                if (parent->getAttr(multiClusterStrategy)) {
                    if (isStrategySOKLike(parent)) {
                        // Process edge case for stop point: last input spilling
                        auto strategyAttr =
                                VPU::MultiClusterStrategyAttr::get(op->getContext(), strategyWithCost.first);
                        auto lastSpillingCost = _layerCostModel.calculateSpillingCost(
                                parent, op, parent->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>(),
                                strategyAttr);
                        rollbackCost += (lastSpillingCost.readCost + lastSpillingCost.writeCost);

                        auto origStrategyAttr = VPU::MultiClusterStrategyAttr::get(op->getContext(), greedyMCStrategy);
                        auto lastSpillingCostForOrig = _layerCostModel.calculateSpillingCost(
                                parent, op, parent->getAttr(multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>(),
                                origStrategyAttr);
                        currentCost += (lastSpillingCostForOrig.readCost + lastSpillingCostForOrig.writeCost);
                    } else {
                        // Case SOH -> SOK/Clustering still need to be processed at next iteration
                        parents.push(parent);
                        _log.trace("    Push parent '{0} to parents queue", parent->getLoc());
                    }
                }
            }
        } else {
            // Process edge case for stop point: last inpust spilling
            auto lastSpillingCost = _layerCostModel.getInputSpillingCost(op, strategyWithCost.first);
            rollbackCost += lastSpillingCost;

            auto lastSpillingCostForOrig = _layerCostModel.getInputSpillingCost(op, greedyMCStrategy);
            currentCost += lastSpillingCostForOrig;
        }
    }

    if (rollbackCost < currentCost) {
        for (auto& parent : parentsToChange) {
            auto parentOp = parent.first;
            auto newStrategy = parent.second;
            parentOp->setAttr(multiClusterStrategy,
                              VPU::MultiClusterStrategyAttr::get(parentOp->getContext(), newStrategy));
            _log.trace("  [parent] '{0}' : set strategy as {1}", parentOp->getLoc(), newStrategy);
        }
        for (auto& child : childrenToChange) {
            auto childOp = child.first;
            auto newStrategy = child.second;
            childOp->setAttr(multiClusterStrategy,
                             VPU::MultiClusterStrategyAttr::get(childOp->getContext(), newStrategy));
            _log.trace("  [child] '{0}' : set strategy as {1}", childOp->getLoc(), newStrategy);
        }
        _log.trace("Subgraph opt: rollback success! New strategies accepted");
    } else {
        _log.trace("Subgraph opt: rollback unneccessary! Strategies no change");
    }
    _log.trace("  Rollback cost: {0} , Current cost: {1}", rollbackCost, currentCost);
}

void SubgraphOptimizer::optimizeStrategyAvoidSpillingOnModel() {
    const auto callback = [this](VPU::NCEOpInterface nceOp) {
        optimizeStrategyAvoidSpillingOnSubgraph(nceOp);
    };

    /// @brief Traversing nodes with preOrder to execute subgraph optimization
    _func.walk(callback);
}
