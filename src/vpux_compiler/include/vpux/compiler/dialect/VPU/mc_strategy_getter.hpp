//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/mc_strategy_getter_interface.hpp"

namespace vpux::VPU {

/*
   Class for getting common strategies for all platforms
*/
class StrategyGetterCommon : public IStrategyGetter {
public:
    virtual void getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const override;
};

}  // namespace vpux::VPU
