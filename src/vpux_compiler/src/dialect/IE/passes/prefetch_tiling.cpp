//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {
//
// PrefetchTiling
//

class PrefetchTiling final : public mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface> {
public:
    PrefetchTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("PrefetchTiling");
    }
    mlir::LogicalResult matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PrefetchTiling::matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    // There are two types of strategy for overlapping DPU and DMA
    // 1. Prefetching - overlapping the child's first weights
    // read with the parent's last compute tile.
    // 2. Pipelining - ensuring the child's second weights
    // read can overlap with its own first compute.
    // Prefetching is addressed in this pass/
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    const auto resShape = getShape(op->getResult(0));
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    if (tilingInfo.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, _log.nest())) {
        // The current op fits into CMX
        // Increase the tiling number to make it prefetchable to its parent op
        auto tiles = vpux::IE::getTilingStrategy(op, _log.nest(), TilingMode::PATTERN_PREFETCH);
        _log.nest(1).trace("Create {0} tiles:", tiles.size());
        return applyTileStrategy(origOp, tiles, rewriter, _log);
    } else {
        const auto tiles = vpux::IE::getTilingStrategy(op, _log.nest(), TilingMode::PREFETCH);
        _log.nest(1).trace("Create {0} tiles:", tiles.size());
        return applyTileStrategy(origOp, tiles, rewriter, _log);
    }
    return mlir::success();
}

//
// PrefetchTilingPass
//
class PrefetchTilingPass final : public IE::PrefetchTilingBase<PrefetchTilingPass> {
public:
    explicit PrefetchTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void PrefetchTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp, IE::ConcatOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp>();
    target.markOpRecursivelyLegal<VPU::NCEClusterTilingOp>([&](mlir::Operation*) {
        return true;
    });
    target.markUnknownOpDynamicallyLegal([this](mlir::Operation* op) {
        if (op->hasAttr(manualTilingStrategyApplied)) {
            return true;
        }
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
            const auto resShape = getShape(op->getResult(0));
            if (!iface.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, _log.nest())) {
                return false;
            }
            if (vpux::IE::prefetchTilingConditionSatisfied(op, _log)) {
                return false;
            }
        }

        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PrefetchTiling>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPrefetchTilingPass(Logger log) {
    return std::make_unique<PrefetchTilingPass>(log);
}
