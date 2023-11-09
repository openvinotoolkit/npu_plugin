//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

using namespace vpux;

namespace {

//
// RollBackTilingStrategyPass
//
class RollBackTilingStrategyPass final : public VPU::RollBackTilingStrategyBase<RollBackTilingStrategyPass> {
public:
    explicit RollBackTilingStrategyPass(bool enablePrefetchTiling, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult RollBackTilingStrategyPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading createRollBackTilingStrategyPass argument by MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

void RollBackTilingStrategyPass::safeRunOnFunc() {
    auto func = getOperation();
    // Recalculate the tiling strategy for the unrolled operations
    func->walk([&](VPU::TilingBuilderOpInterface origOp) {
        if (!mlir::isa<VPU::VerticalFusionOpInterface>(origOp.getOperation())) {
            return;
        }

        if (origOp->getParentOfType<VPU::VerticalFusionOp>() != nullptr) {
            return;
        }

        if (!VPU::opNeedsTiling(origOp, _enablePrefetchTiling, _log)) {
            return;
        }

        auto origTilingStrategy = getLayerTilingStrategy(origOp, _enablePrefetchTiling, _log);
        if (mlir::failed(origTilingStrategy)) {
            _log.trace("No valid tiling strategy for {0} {1}", origOp->getName(), origOp->getLoc());
            return;
        }
        const auto origTilingStrategyAttr =
                getIntArrayAttr(origOp->getContext(), origTilingStrategy.getValue()[0].axis);
        const auto curTilingStrategy = origOp->getAttr(tilingStrategy).cast<mlir::ArrayAttr>();

        if (curTilingStrategy != origTilingStrategyAttr) {
            _log.nest().trace("Roll back tiling strategy from {0} to {1}", curTilingStrategy, origTilingStrategyAttr);
            origOp->setAttr(tilingStrategy, origTilingStrategyAttr);
        }
    });
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRollBackTilingStrategyPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<RollBackTilingStrategyPass>(enablePrefetchTiling, log);
}
