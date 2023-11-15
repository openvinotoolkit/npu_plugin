//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/mc_strategy_getter.hpp"

namespace vpux::VPU {

/*
   Class for getting strategies for VPU30XX
*/
class StrategyGetterVPU30XX : public StrategyGetterCommon {
public:
    virtual void getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const override {
        StrategyGetterCommon::getMCStrategies(strategies);
        strategies.push_back(MultiClusterStrategy::SplitOverHeight);
        strategies.push_back(MultiClusterStrategy::SplitOverKernel);
        strategies.push_back(MultiClusterStrategy::HKSwitch);
        strategies.push_back(MultiClusterStrategy::SplitOverHeightOverlapped);
    }
};

}  // namespace vpux::VPU
