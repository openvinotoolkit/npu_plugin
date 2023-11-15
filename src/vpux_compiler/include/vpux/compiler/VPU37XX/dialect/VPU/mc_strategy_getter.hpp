//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/VPU30XX/dialect/VPU/mc_strategy_getter.hpp"

namespace vpux::VPU {

/*
   Class for getting strategies for VPU37XX
*/
class StrategyGetterVPU37XX : public StrategyGetterVPU30XX {};

}  // namespace vpux::VPU
