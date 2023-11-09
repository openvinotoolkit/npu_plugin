//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <llvm/ADT/BitVector.h>

#include "strategy.hpp"

#include <unordered_map>

#include <vpu/cycles_interface_types.h>

namespace vpux::VPU {

/*
   Type of cost of strategies
*/
using StrategyCost = VPUNN::CyclesInterfaceType;

/*
   Bits identify states
   0 - current state
   1 - best state
   if none of them are specified, strategy neither current or best
*/
using StrategyState = llvm::BitVector;

/*
   Link between operation and one of its strategy
*/
using OperationStrategy = std::pair<mlir::Operation*, Strategy>;

/*
   Unified info about particular strategy
   Strategy itself, its cost, its state
*/
struct StrategyInfo {
    StrategyInfo(const Strategy& str, const StrategyCost cost, const StrategyState& state)
            : strategy(str), strategyCost(cost), strategyState(state) {
    }

    Strategy strategy;
    StrategyCost strategyCost;
    StrategyState strategyState;
};

/*
   The key to store transition cost between two
   operations and their strategies
*/
struct CombinedTransitionKey {
    CombinedTransitionKey(mlir::Operation* srcOp, const Strategy& srcStr, mlir::Operation* dstOp,
                          const Strategy& dstStr)
            : srcOperation(srcOp), srcStrategy(srcStr), dstOperation(dstOp), dstStrategy(dstStr) {
    }

    bool operator==(const CombinedTransitionKey& other) const {
        return (srcOperation == other.srcOperation && srcStrategy == other.srcStrategy &&
                dstOperation == other.dstOperation && dstStrategy == other.dstStrategy);
    }

    mlir::Operation* srcOperation;
    Strategy srcStrategy;
    mlir::Operation* dstOperation;
    Strategy dstStrategy;
};

/*
   Hash function for Combined Transition key
*/
struct hashCombinedKey {
    std::size_t operator()(const CombinedTransitionKey& key) const {
        auto srcTilingStr = key.srcStrategy.getTilingStrategy();
        auto dstTilingStr = key.dstStrategy.getTilingStrategy();

        return std::hash<mlir::Operation*>()(key.srcOperation) ^
               std::hash<VPU::MultiClusterStrategy>()(key.srcStrategy.getMCStrategy()) ^
               (srcTilingStr != nullptr ? std::hash<ArrayRef<mlir::Attribute>>()(srcTilingStr.getValue())
                                        : std::hash<std::nullptr_t>()(nullptr)) ^
               std::hash<mlir::Operation*>()(key.dstOperation) ^
               std::hash<VPU::MultiClusterStrategy>()(key.dstStrategy.getMCStrategy()) ^
               (dstTilingStr != nullptr ? std::hash<ArrayRef<mlir::Attribute>>()(dstTilingStr.getValue())
                                        : std::hash<std::nullptr_t>()(nullptr));
    }
};

/*
    OperationStrategies
    Storage for link between operation and it's possible strategies.
    For each strategy cost and state are specified, for instance

    Conv -> strategy = {SOH, no tiling}, cost = 200, best solution
         -> strategy = {SOK, tiling over H}, cost = 300, current solution

    and also cost of transition between 2 operations when some strategies
    are chosen for each of them
    for instance, these two transitions might have different cost when strategy is
    changed for one of operation.
    "Operation 1, stategy 1 ->  Operation 2, strategy 1"
    "Operation 1, stategy 2 ->  Operation 2, strategy 1"
*/

class OperationStrategies final {
public:
    OperationStrategies(){};

    /*
       Add new strategy for operation with cost
       State is not specified by default
       It throws exception if strategy was already added
    */
    void addStrategy(const OperationStrategy& opStr, const StrategyCost cost);

    /*
       Update cost for existed strategy
       It throws exception is strategy is not
       in the list for operation
    */
    void setStrategy(const OperationStrategy& opStr, const StrategyCost cost);

    /*
       Checks if strategy is set for operation
    */
    bool hasStrategy(const OperationStrategy& opStr) const;

    /*
       Checks if any strategy is set for operation
    */
    bool hasAnyStrategy(mlir::Operation* op) const;

    /*
       Set state to "current" for pair operation -> strategy
    */
    void setCurrentStrategy(const OperationStrategy& opStr);

    /*
       Set state to "best" for pair operation -> strategy
    */
    void setBestStrategy(const OperationStrategy& opStr);

    /*
       Set cost of transition between two operation -> strategy links
       In case there is already the cost, it's updated
    */
    void setTransitionCost(const OperationStrategy& srcOpStr, const OperationStrategy& dstOpStr,
                           const StrategyCost cost);

    /*
       Get cost of particular strategy for operation
    */
    StrategyCost getStrategyCost(const OperationStrategy& opStr) const;

    /*
       Get transition cost between two operation with strategies.
       If the cost hasn't been set, function returns None
    */
    mlir::Optional<StrategyCost> getTransitionCost(const OperationStrategy& srcOpStr,
                                                   const OperationStrategy& dstOpStr) const;

    /*
       Get "current" strategy for operation
    */
    Strategy getCurrentStrategy(mlir::Operation* operation) const;

    /*
       Get "best" strategy for operation
    */
    Strategy getBestStrategy(mlir::Operation* operation) const;

private:
    /*
     * Current number of bits for StrategyState
     */
    static constexpr unsigned BITS_NUMBER = 2;
    static constexpr unsigned CURRENT_STATE_INDEX = 0;
    static constexpr unsigned BEST_STATE_INDEX = 1;

    /*
       Set state for link operation -> strategy
    */
    void setStrategyState(const OperationStrategy& opStr, unsigned bitIndex);

    /*
       Get strategy of operation by state (current, best)
    */
    Strategy getStrategyByState(mlir::Operation* operation, unsigned bitIndex) const;

    /*
       Calculate hash key to store transition cost between two operations
    */
    CombinedTransitionKey getTransitionHash(const OperationStrategy& srcOpStr, const OperationStrategy& dstOpStr) const;

    std::unordered_map<mlir::Operation*, SmallVector<StrategyInfo>> _strategies;
    std::unordered_map<CombinedTransitionKey, StrategyCost, hashCombinedKey> _transitionCost;
};

}  // namespace vpux::VPU
