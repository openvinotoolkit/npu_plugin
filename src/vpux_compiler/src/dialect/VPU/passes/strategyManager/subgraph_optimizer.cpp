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

double bestCostOfKCompatible(mlir::Operation* op, bool allowHK){
    double SOKCost = greedyCostOfLayer(op, splitOverKernel);
    double clusteringCost = greedyCostOfLayer(op, clustering);
    double HKSwitchCost = allowHK? greedyCostOfLayer(op, HKSwitch) : COST_MAX;
    return std::min({SOKCost, clusteringCost, HKSwitchCost});
}

// @warning I drop the allowHK flag for HCompatible
double bestCostOfHCompatible(mlir::Operation* op){
    double SOHCost = greedyCostOfLayer(op, splitOverHeight);
    double SOHOverCost = greedyCostOfLayer(op, splitOverHeightOverlapped);
    return std::min({SOHCost, SOHOverCost});
}

// @mcm let's don't consider the spill portion of the SOH->SOK transition point here
// just consider, from a pure computational time persepective, will this layer be
// better suited to SOH or SOK
// @warning Currently we may not need this in vpux, as we didn't consider spilling 
// in our simple cost model
bool hasGreedySOK(mlir::Operation* op){
    auto hCost = bestCostOfHCompatible(op);
    auto kCost = bestCostOfKCompatible(op, true);
    return kCost < hCost ? true : false;
}

void SubgraphOptimizer::rollbackOrSpill(){
    const auto callback = [](mlir::Operation* op){
        if(!op->getAttr(multiClusterStrategy)) continue;

    }

    _func.walk(callback);

}