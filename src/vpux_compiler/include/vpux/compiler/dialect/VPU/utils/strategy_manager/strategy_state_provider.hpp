//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "state_provider_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"

#include <llvm/ADT/SetVector.h>

#include <map>
#include <queue>
#include <random>

namespace vpux::VPU {

class LayerVPUNNCost;

/*
   Implementation of functions to get state/cost for optimization algorithm
*/
class DefaultStateProvider : public IStateProvider {
public:
    DefaultStateProvider(const std::shared_ptr<OperationStrategies>& storage,
                         const std::shared_ptr<LayerVPUNNCost>& costModel)
            : _storage(storage), _costModel(costModel), _generator(0) {
    }

    /*
      Get operation and one of its associated strategies
    */
    OperationStrategy getState(int temperature, double& cost, const OperationStrategy* const state) override;

    /*
      Get cost associated with selected pair of operation -> strategy
    */
    StrategyCost getCost(const OperationStrategy& state) override;

    /*
      Update current state for pair operation -> strategy
    */
    void updateState(const OperationStrategy& state) override;

    /*
      Update best state for pair operation -> strategy
    */
    void updateSolution(const OperationStrategy& state) override;

    /*
      Get full cost for IR
    */
    StrategyCost getFullCost() override;

private:
    /*
       Choose randomly operation from the list and return its current strategy
    */
    OperationStrategy randomOperation(mlir::ArrayRef<mlir::Operation*> operations);

    /*
      Get transition cost between two states of operations
    */
    StrategyCost getTransitionCost(const OperationStrategy& firstState, const OperationStrategy& secondState);

    /*
      Get transition cost between operation from storage and operation outside storage
    */
    StrategyCost getTransitionOutsideCost(const OperationStrategy& state, mlir::Operation* operation,
                                          const bool parent);

    /*
      Prepare parameters to call cost model
    */
    VPUNNCostParameters getCostModelParameters(const OperationStrategy& state) const;

    /*
      Get parent operation of one of input
    */
    mlir::Operation* getParentOp(mlir::Value operand) const;

    /*
      Get user operation of one of input
    */
    void getConsumersOp(SmallVector<mlir::Operation*>& users, mlir::Operation* op) const;

    /*
      Check if operation might be part of tiling
    */
    bool hasTiling(mlir::Operation* operation) const;

    /*
      Check if there is no spill between operations
    */
    bool canStayInCMX(const OperationStrategy& firstState, const OperationStrategy& secondState) const;

    bool spillAroundConcat(mlir::Operation* operation) const;

    /*
      Check if there is no spill between strategies
    */
    bool doMCStrategiesMatch(const MultiClusterStrategy parentStrategy, const MultiClusterStrategy childStrategy) const;

    /*
      Check if tiling of operation allows to stay in CMX
    */
    bool isCMXConcatentationAvaliable(mlir::Operation* operation, const TilingMode mode, const OutputTiling& tiles,
                                      const MultiClusterStrategy strategy) const;

    /*
      Summarize transition cost between operation in the state and its neighbours
    */
    StrategyCost accumulateCost(ArrayRef<mlir::Operation*> neighbours, const OperationStrategy& state,
                                bool parent = true);

    /*
      Get all neighbours for operation
    */
    void fillInNeighbours(mlir::Operation* operation);

    /*
      Get back to best solution
    */
    void reannealingStep(int temperature, double& cost);

    /*
      Steps to set up values based on initial temperature
    */
    void initializeTemperature(int temperature);

    /*
      Storage with data for optimization
    */
    std::shared_ptr<OperationStrategies> _storage;

    /*
      VPUNN cost del for getting cost for operations and DMAs
    */
    std::shared_ptr<LayerVPUNNCost> _costModel;

    /*
      Store neighbours for each operation
    */
    std::unordered_map<mlir::Operation*, std::pair<SmallVector<mlir::Operation*>, SmallVector<mlir::Operation*>>>
            _neighbours;

    /*
      Points where storage is got back to best recent states
    */
    std::queue<int> _reannealingTemperatures;

    /*
      Common initial temperature
    */
    std::optional<int> _initialTemperature;

    /*
      Random generator
    */
    std::mt19937 _generator;
};

}  // namespace vpux::VPU
