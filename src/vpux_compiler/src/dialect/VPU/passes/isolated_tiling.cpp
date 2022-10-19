//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/FunctionExtras.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <numeric>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// GenericTiling
//

class GenericTiling final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    GenericTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("GenericTiling");
    }

    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GenericTiling::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto tiles = VPU::getTilingStrategy(origOp, _log);
    _log.nest(1).trace("Create {0} tiles:", tiles.size());

    return VPU::applyTileStrategy(origOp, tiles, rewriter, _log);
}

//
// IsolatedTilingPass
//

class IsolatedTilingPass final : public VPU::IsolatedTilingBase<IsolatedTilingPass> {
public:
    explicit IsolatedTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void IsolatedTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp>();
    target.markOpRecursivelyLegal<VPU::NCEClusterTilingOp>([&](mlir::Operation*) {
        return true;
    });
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op) && !archSupportsSwLayerTiling(arch)) {
            return true;
        }

        if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
            const auto resShape = getShape(op->getResult(0));
            return iface.isSupportedTiling({TileInfo(resShape)}, TilingMode::ISOLATED, _log.nest());
        }

        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericTiling>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createIsolatedTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createIsolatedTilingPass(Logger log) {
    return std::make_unique<IsolatedTilingPass>(log);
}
