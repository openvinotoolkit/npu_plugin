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

#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

SubgraphOptimizer::SubgraphOptimizer(mlir::FuncOp func, Logger log):
    _func(func), _log(log){
}

void SubgraphOptimizer::verifySpillStrategies(bool lockClusteringStrategy=false){

}

bool SubgraphOptimizer::addSpillsAtStrategyTransition(){

}

// Op is Kcompatible means the following op can be SOK
bool SubgraphOptimizer::isKCompatible(mlir::Operation* op, bool allowHK){
    const auto strategy = op->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue();
    if (strategy == splitOverKernel || strategy == clustering || (allowHK && strategy == HKSwitch)){
        return true;
    }
    return false;
}

// @tbd decided by bdk5
bool SubgraphOptimizer::canBeSOK(mlir::Operation* op){
    if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>
        || mlir::isa<VPU::NCEMaxPoolOp>(op)) {
            return true;
    }
    return false;
}

bool SubgraphOptimizer::canBeHKSwitch(mlir::Operation* op){
    // @todo according to op type: conv/dwConv/maxpool -> hk
    if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>
        || mlir::isa<VPU::NCEMaxPoolOp>(op)) {
            return true;
    }
    return false;
}

void SubgraphOptimizer::skipSOH(mlir::Operation* op, bool allowHK){
    auto opID = op->getName().getStringRef();
    if (!_strategySkip.find(opID)){
        _strategySkip.insert({opID, {}});
    }

    if (_strategySkip.at(opID).find(splitOverHeight)){
        _strategySkip.at(opID).at(splitOverHeight) = true;
    }else{
        _strategySkip.at(opID).insert({splitOverHeight, true})
    }

    if (_strategySkip.at(opID).find(splitOverHeightOverlapped)){
        _strategySkip.at(opID).at(splitOverHeightOverlapped) = true;
    }else{
        _strategySkip.at(opID).insert({splitOverHeightOverlapped, true})
    }

    if (!allowHK){
        if (_strategySkip.at(opID).find(HKSwitch)){
            _strategySkip.at(opID).at(HKSwitch) = true;
        }else{
            _strategySkip.at(opID).insert({HKSwitch, true})
        }
    }
}

bool SubgraphOptimizer::isSpillingFromStrategy(mlir::Operation* op){
    bool opKCompatible = isKCompatible(op);
    for (auto child : op->getResult(0).getUsers()){
        if (!child->getAttr(multiClusterStrategy)) continue;
        if (opKCompatible != isKCompatible(child)){
            return true;
        }
    }
    return false;
}

double SubgraphOptimizer::bestCostOfKCompatible(mlir::Operation* op, bool allowHK){
    double SOKCost = greedyCostOfLayer(op, splitOverKernel);
    double clusteringCost = greedyCostOfLayer(op, clustering);
    double HKSwitchCost = allowHK? greedyCostOfLayer(op, HKSwitch) : COST_MAX;
    return std::min({SOKCost, clusteringCost, HKSwitchCost});
}

// @warning I drop the allowHK flag for HCompatible
double SubgraphOptimizer::bestCostOfHCompatible(mlir::Operation* op){
    double SOHCost = greedyCostOfLayer(op, splitOverHeight);
    double SOHOverCost = greedyCostOfLayer(op, splitOverHeightOverlapped);
    return std::min({SOHCost, SOHOverCost});
}

// @mcm let's don't consider the spill portion of the SOH->SOK transition point here
// just consider, from a pure computational time persepective, will this layer be
// better suited to SOH or SOK
// @warning Currently we may not need this in vpux, as we didn't consider spilling 
// in our simple cost model
bool SubgraphOptimizer::hasGreedyKCompatible(mlir::Operation* op){
    auto hCost = bestCostOfHCompatible(op);
    auto kCost = bestCostOfKCompatible(op, true);
    return kCost < hCost ? true : false;
}

bool SubgraphOptimizer::areChildrenKCompatible(mlir::Operation* op){
    for (auto child : op->getResult(0).getUsers()){
        if (child->getAttr(multiClusterStrategy) && !isKCompatible(child)){
            return false;
        }
    }
    return true;
}

void SubgraphOptimizer::setKCompatible(mlir::Operation* op, bool allowHK=true){
    if (allowHk){
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "HKSwitch"));
    }else if (greedyCostOfLayer(op, splitOverHeight) < greedyCostOfLayer(op, clustering)){
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "SplitOverKernel"));
    }else{
        op->setAttr(multiClusterStrategy, mlir::StringAttr::get(origOp->getContext(), "Clustering"));
    }
}

void SubgraphOptimizer::singleRollback(mlir::Operation* op){
    setKCompatible(op, true);
    skipSOH(op, true);
}

// @mcm The idea of this algorithm is to decide between spilling to change tensor split strategy,
// rolling back SOH to some point where this strategy switch can happen in CMX (HKSwitch)
// We move backwards through a topological sort of the Op Model
// For each op, if it is a strategy spill we first decide what ops would need to change for a rollback
// Consider a simple linear graph A (SOH or HK) -> B (SOH) -> C (SOH) -> D (SOK)
// When we reach C, a SOH layer that must spill we process it:
// 1. Add C to ops_to_change, mark it
// 2. Add unmarked children to Q_c, if they have SOH or HKSwitch strategy and mark them
//      ex: Q_c : remains empty
// 3. If C cannnot take HKSwitch, add unmarked parents to Q_p and mark them
//      ex: Q_p : B
// 4. Continue processing elements from Q_p while not empty
//      ex: pop B and process it from step 1 (Q_c will remain empty, Q_p: A)
//          pop A and process it from step 1 (Q_c will remain empty, Q_p is empty)
// 5. Continue processing elements from Q_c while not empty
// 6. If cost cheaper to roll back, last HK-elligble op added to ops to change is HK,
//    rest take best compatible (SOK, clus) strategy
void SubgraphOptimizer::rollbackOnSubgraph(mlir::Operation* op){
    double currentCost = 0.0;
    double rollbackCost = 0.0;
    std::queue<mlir::Operation*> parents;
    std::queue<mlir::Operation*> children;
    std::vector<mlir::Operation*> parentsToChange;
    std::vector<mlir::Operation*> childrenToChange;
    std::set<std::string> processedOps;
    // @todo Need confirm that
    // auto heuristicMultiplier = getMultiplier(opIt);

    // @tbd forceRollback() need confirm

    // Only process node with greedy strategy SOH
    if (!isSpillingFromStrategy(op) || isKCompatible(op, true)) continue;
    
    // SOH -> KCompatible
    parents.push_back(op);    
    while(!parents.empty() || !children.empty()){
        mlir::Operation *opIt= nullptr;
        if (!parents.empty()){
            opIt= parents.front();
            parents.pop();
            parentsToChange.push_back(opIt);
            rollbackCost += bestCostOfKCompatible(opIt);
        }else{
            opIt= children.front();
            children.pop();
            childrenToChange.push_back(opIt);
            rollbackCost += bestCostOfKCompatible(opIt);
        }

        // Had better record the cost for each layer and read it
        currentCost+= greedyCostOfLayer(opIt, opIt->getAttr(multiClusterStrategy).cast<mlir::StringAttr>().getValue());
        processedOps.insert(opIt->getName());

        // Add children if they are still in SOH compatible
        for (auto child : op->getResult(0).getUsers()){
            if (processedOps.find(child->getName())) continue;
            if (auto childStrategy= child->getAttr(multiClusterStrategy)){
                auto childStrategy_ = childStrategy.cast<mlir::StringAttr>().getValue();
                if (childStrategy_ == SplitOverKernel || childStrategy_ == HKSwitch){
                    children.push(child);
                }
            } 
        }

        // We stop moving up the graph when we find nodes that could be HK switch points
        // or nodes that are already in compatible strategies (SOK, Clus)
        for (auto input : opIt.inputs()) {
            auto parent = input.getDefiningOp();
            if (auto parentStrategy= parent->getAttr(multiClusterStrategy)){
                if ((canBeSOK(op) && isKCompatible(parent)) || (canBeHKSwitch(op) && !isKCompatible(parent)))
                    continue;
                parents.push(parent);
            }
        }
    }

    if (rollbackCost < currentCost){
        for (auto op : parentsToChange){
            setKCompatible(op, true);
        }
        for (auto op : childrenToChange){
            setKCompatible(op, false);
        }
    }
}

// @mcm This function is the heart of what the StrategyManager and MetaGraph did for the
// Graph Optimizer pass. The idea is, when strategies on multiple ops must be changed
// together (i.e. making choices around SOH, SOK), this pass will decide the most efficient way
// to do those strategy transitions.
// The options are, 1. rollback the transition to some earlier point in the graph
// 2. Spill to do the transition on the spot
// 3. If possible, do the transition in CMX
// To decide between these options, we look at each op in turn moving backwards through the op model
// If it marks a transition point of SOH->SOK, then we search the graph for all its neighbors that would
// also require a change if this op where to change to K-compatible. As we go, we tally the cost of changing
// each op, and at the end we compare that to the current cost of this neighbor subgraph. We choose the more
// performant strategy, which either requires processing the subgraph to change each node, or leaving it be.
void SubgraphOptimizer::rollbackOrSpill(){
    const auto callback = [](mlir::Operation* op){
        if(!op->getAttr(multiClusterStrategy)) continue;

        if(isSpillingFromStrategy(op) && hasGreedyKCompatible(op) && areChildrenKCompatible(op)){
            singleRollback(op);
        }else{
            rollbackOnSubgraph(op);
        }

    }

    _func.walk(callback);
}
