//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/mc_strategy_getter.hpp"

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<StrategyGetterBase> createMCStrategyGetter(ArchKind arch, int64_t numClusters);

}  // namespace vpux::VPU
