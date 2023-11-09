//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/mc_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/mc_strategy_getter_factory.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"

#include "common/utils.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_MCStrategy_Getter = MLIR_UnitBase;

TEST_F(MLIR_MCStrategy_Getter, MCGetterList) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto numClusters = 2;

    SmallVector<VPU::MultiClusterStrategy> strategyVPU37XXSet;
    auto mcGetter = VPU::createMCStrategyGetter(VPU::ArchKind::VPUX37XX, numClusters);

    mcGetter->getMCStrategies(strategyVPU37XXSet);
    EXPECT_EQ(strategyVPU37XXSet.size(), 5);

    SmallVector<VPU::MultiClusterStrategy> strategyVPU37XX1TileSet;
    mcGetter = VPU::createMCStrategyGetter(VPU::ArchKind::VPUX37XX, 1);

    mcGetter->getMCStrategies(strategyVPU37XX1TileSet);
    EXPECT_EQ(strategyVPU37XX1TileSet.size(), 1);
}
