//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/mc_strategy_getter_interface.hpp"

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<IStrategyGetter> createMCStrategyGetter(ArchKind arch, int64_t numClusters);

}  // namespace vpux::VPU
