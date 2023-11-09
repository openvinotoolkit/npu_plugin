//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"

namespace vpux::VPU {

/*
   Interface provides method for getting available strategies
*/
class IStrategyGetter {
public:
    virtual ~IStrategyGetter() = default;
    virtual void getMCStrategies(SmallVector<MultiClusterStrategy>& strategies) const = 0;
};

}  // namespace vpux::VPU
