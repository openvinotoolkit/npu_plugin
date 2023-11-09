//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/mc_strategy_getter.hpp"

using namespace vpux::VPU;

void StrategyGetterCommon::getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const {
    strategies.push_back(MultiClusterStrategy::Clustering);
}
