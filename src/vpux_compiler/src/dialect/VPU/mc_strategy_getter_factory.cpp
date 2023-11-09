//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/mc_strategy_getter_factory.hpp"

#include "vpux/compiler/VPU30XX/dialect/VPU/mc_strategy_getter.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPU/mc_strategy_getter.hpp"

using namespace vpux::VPU;

std::unique_ptr<IStrategyGetter> vpux::VPU::createMCStrategyGetter(ArchKind arch, int64_t numClusters) {
    if (numClusters == 1) {
        return std::make_unique<StrategyGetterCommon>();
    }

    switch (arch) {
    case ArchKind::VPUX30XX: {
        return std::make_unique<StrategyGetterVPU30XX>();
    }
    case ArchKind::VPUX37XX: {
        return std::make_unique<StrategyGetterVPU37XX>();
    }
    case ArchKind::UNKNOWN:
    default: {
        return std::make_unique<StrategyGetterCommon>();
    }
    }
}
