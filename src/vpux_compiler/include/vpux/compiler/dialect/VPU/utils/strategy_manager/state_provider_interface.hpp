//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"

namespace vpux::VPU {

/*
Interface for State Provider.
Each State is a combination of operation and its associated strategy.
In State Provider, for each set of pair of operation and strategy, we select a random operation and  it's strategy,
calculate cost associated with that, update the "state" and update the "best state".
*/
class IStateProvider {
public:
    /*
    Get operation and one of its associated strategies
    */
    virtual OperationStrategy getState(int /*temperature*/, double& cost, const OperationStrategy* const state) = 0;

    /*
    Get cost associated with selected pair of operation -> strategy
    */
    virtual StrategyCost getCost(const OperationStrategy& state) = 0;

    /*
    Update current state for pair operation -> strategy
    */
    virtual void updateState(const OperationStrategy& state) = 0;

    /*
    Update best state for pair operation -> strategy
    */
    virtual void updateSolution(const OperationStrategy& state) = 0;

    /*
    Get full cost for IR
    */
    virtual StrategyCost getFullCost() = 0;

    virtual ~IStateProvider() = default;
};

}  // namespace vpux::VPU
