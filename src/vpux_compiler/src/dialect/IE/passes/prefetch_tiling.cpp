//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <mlir/IR/BlockAndValueMapping.h>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

OutputTiling generatePrefetchTiles(mlir::Operation* op, Logger log) {
    log.trace("Generating prefetch tiles for op {0} at {1}", op->getName(), op->getLoc());

    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    const auto outputShape = getShape(op->getResult(0).getType().cast<mlir::ShapedType>());
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());
    auto getDimsToTile = [](const Shape& nTilesOnDim) -> SmallVector<Dim> {
        SmallVector<Dim> res;
        for (unsigned i = 0; i < nTilesOnDim.size(); i++) {
            if (nTilesOnDim[Dim(i)] > 1)
                res.emplace_back(Dim(i));
        }
        return res;
    };

    // step 1: compute a general tiling strategy to fit into the CMX
    Shape nTilesOnDim = IE::computeGeneralTileStrategy(op, log);
    auto dimsToTile = getDimsToTile(nTilesOnDim);
    VPUX_THROW_WHEN(dimsToTile.size() == 0, "Must tile at least on one dimension");
    if (dimsToTile.size() > 1) {
        // return general tiling when getting nested tiles.
        return fillDividedTiles(nTilesOnDim, outputShape);
    }

    // step 2: increase the general tile strategy to satisfy prefetching
    const auto targetDim = dimsToTile[0];
    Shape prefetchableTilesOnDim = nTilesOnDim;
    while (prefetchableTilesOnDim[targetDim] < 3 * nTilesOnDim[targetDim] &&
           !tilingInfo.isSupportedPrefetchTiling(prefetchableTilesOnDim, log)) {
        // The "3" here is an experimental number from MCM activation prefetch pass.
        // The purpose is to avoid excessive tiling.
        prefetchableTilesOnDim[targetDim]++;
    }

    return tilingInfo.isSupportedPrefetchTiling(prefetchableTilesOnDim, log)
                   ? fillDividedTiles(prefetchableTilesOnDim, outputShape)
                   : fillDividedTiles(nTilesOnDim, outputShape);
}

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
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto tiles = generatePrefetchTiles(origOp.getOperation(), _log.nest());
    _log.nest(1).trace("Create {0} tiles:", tiles.size());

    return applyTileStrategy(origOp, tiles, rewriter, _log);
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
    target.markUnknownOpDynamicallyLegal([this](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
            const auto resShape = getShape(op->getResult(0));
            return iface.isSupportedTiling({TileInfo(resShape)}, _log.nest());
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
