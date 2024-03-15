//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"

namespace vpux::VPU {

/*
Interface for Strategy Optimization Algorithm.
We get the state from State Provider and optimize the operation strategy.
 */
class IStrategyOptAlgorithm {
public:
    /*
    Optimize operation strategy set using simulated annealing.
     */
    virtual void optimize() = 0;

    virtual ~IStrategyOptAlgorithm() = default;
};

}  // namespace vpux::VPU
