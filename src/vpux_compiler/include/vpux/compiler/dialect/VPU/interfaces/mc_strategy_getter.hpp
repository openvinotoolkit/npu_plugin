//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::VPU {

/*
   Class for getting available strategies
*/
class StrategyGetterBase {
public:
    virtual ~StrategyGetterBase() = default;

    virtual void getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const;
};

}  // namespace vpux::VPU
