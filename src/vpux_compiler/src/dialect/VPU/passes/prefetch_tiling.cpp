//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {
//
// PrefetchTiling
//
// There are three modes of tiling strategies as defined in vpux::TilingMode:
// 1. ISOLATED tiling: Split operations with smallest tiling number to make them fit into CMX
// 2. PIPELINING tiling: Overlap the DPU time of earlier sub-tile with the DMA time of the later ones
//              Two possible scenarios where the PIPELINING could be triggered:
//              a). When ISOLATED tiling is required, the tiling number will be increased to satisfy PIPELINING
//              b). When the constant weights of an operation is larger than the threshold
//                  tiling number will be increased to satisfy PIPELINING
//                  even though the operation doesn't require ISOLATED tiling originally
//              A precondition is that a feasible tiling strategy must exist to make PIPELINING work
//              Otherwise it will fallback to ISOLATED tiling or non-tiling
// 3. PREFETCHING tiling: Overlap the DPU time of parent operation with the DMA time the child
//

class PrefetchTiling final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    PrefetchTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("PrefetchTiling");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PrefetchTiling::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    const auto resShape = getShape(op->getResult(0));
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    // SW layer tiling
    if (!mlir::isa<VPU::NCEOpInterface>(op)) {
        _log.nest(1).trace("Attempting ISOLATED tiling SW layer.");
        const auto tiles = origOp.getTilingStrategy(vpux::TilingMode::ISOLATED, _log.nest());
        _log.nest(1).trace("ISOLATED tiling: Create {0} tiles:", tiles.size());
        return VPU::applyTileStrategy(origOp, tiles, rewriter, _log.nest());
    }

    // HW layer tiling
    if (tilingInfo.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, _log.nest()) &&
        vpux::VPU::prefetchTilingConditionSatisfied(op, _log.nest())) {
        _log.nest(1).trace("Attempting PREFETCHING tiling for NCE layer.");
        auto tiles = origOp.getTilingStrategy(TilingMode::PREFETCHING, _log.nest());
        _log.nest(1).trace("PREFETCHING tiling: Create {0} tiles:", tiles.size());
        return VPU::applyTileStrategy(origOp, tiles, rewriter, _log.nest());
    } else {
        _log.nest(1).trace("Attempting ISOLATED/PIPELINING tiling NCE layer.");
        const auto tiles = origOp.getTilingStrategy(TilingMode::PIPELINING, _log.nest());
        _log.nest(1).trace("ISOLATED/PIPELINING tiling: Create {0} tiles:", tiles.size());
        return VPU::applyTileStrategy(origOp, tiles, rewriter, _log.nest());
    }
}

//
// PrefetchTilingPass
//
class PrefetchTilingPass final : public VPU::PrefetchTilingBase<PrefetchTilingPass> {
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
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (op->hasAttr(manualTilingStrategyApplied) || op->hasAttr(tilingStrategy)) {
            return true;
        }
        if (!mlir::isa<VPU::NCEOpInterface>(op) && !archSupportsSwLayerTiling(arch)) {
            return true;
        }
        if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
            _log.trace("Check: '{0}' at '{1}'", op->getName(), op->getLoc());
            const auto resShape = getShape(op->getResult(0));
            if (!iface.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, _log.nest())) {
                _log.nest(1).trace("ISOLATED tiling or PIPELINING tiling required");
                return false;
            }
            if (mlir::isa<VPU::NCEOpInterface>(op)) {
                if (vpux::VPU::prefetchTilingConditionSatisfied(op, _log.nest())) {
                    _log.nest(1).trace("PREFETCHING tiling required");
                    return false;
                }
                if (vpux::VPU::largeConstPipelineConditionSatisfied(op, _log.nest())) {
                    _log.nest(1).trace("PIPELINING tiling for large constant weights required");
                    return false;
                }
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

std::unique_ptr<mlir::Pass> vpux::VPU::createPrefetchTilingPass(Logger log) {
    return std::make_unique<PrefetchTilingPass>(log);
}
