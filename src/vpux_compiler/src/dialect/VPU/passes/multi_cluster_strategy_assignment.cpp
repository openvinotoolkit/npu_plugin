//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/strategy_manager.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// MultiClusterStrategyAssignmentPass
//

class MultiClusterStrategyAssignmentPass final :
        public MultiClusterStrategyAssignmentBase<MultiClusterStrategyAssignmentPass> {
public:
    explicit MultiClusterStrategyAssignmentPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnFunc() final;
};

mlir::LogicalResult MultiClusterStrategyAssignmentPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void MultiClusterStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto nceCluster = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    if (nceCluster.count() > 1) {
        LayerStrategyCheckerFactory::instance().registerClusteredOpStrategy(func, _log);

        StrategyManager strategyManager(func, _log.nest());
        _log.trace("Greedy Strategy Assignment");
        strategyManager.assignMultiClusterStrategy();
        _log.trace("Execute Subgraph Optimization");
        strategyManager.optimizeMulticlusterStrategy();
        _log.trace("Remove Temporary Strategy");
        strategyManager.removeTemporaryMulticlusterStrategy();
    }
}

}  // namespace

//
// createMultiClusterStrategyAssignmentPass
//

std::unique_ptr<mlir::Pass> VPU::createMultiClusterStrategyAssignmentPass(Logger log) {
    return std::make_unique<MultiClusterStrategyAssignmentPass>(log);
}
