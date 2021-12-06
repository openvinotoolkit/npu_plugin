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

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"


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
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto tiles = origOp.generatePrefetchTiling(_log.nest());

    _log.nest(1).trace("Create {0} tiles:", tiles.size());
    for (const auto& outputTile : tiles) {
        _log.nest(2).trace("{0}", outputTile);
    }

    SmallVector<mlir::Value> resultTileVals;
    SmallVector<ShapeRef> resultTileOffsets;

    resultTileVals.reserve(tiles.size());
    resultTileOffsets.reserve(tiles.size());

    for (const auto& outputTile : tiles) {
        const auto tiledRes = origOp.reifyTile(outputTile, rewriter);

        const auto tiledShape = getShape(tiledRes);
        VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                          "Inferred tiled output shape '{0}' doesn't match with generated '{1}'", tiledShape,
                          outputTile.shape);

        resultTileVals.push_back(tiledRes);
        resultTileOffsets.push_back(outputTile.offsets);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, origOp->getResult(0).getType(), mlir::ValueRange(resultTileVals),
                                              makeArrayRef(resultTileOffsets));
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

    const auto isUnPrefetchable = [this](mlir::Operation* op) {
        // check the prefetch tiling strategy.
        // only consider conv first
        auto convOp = llvm::dyn_cast<IE::ConvolutionOp>(op);
        if (!convOp)
            return true;

        const auto tileDim = Shape(parseIntArrayAttr<int64_t>(convOp.tiling_strategyAttr()));
        // check prefetchable
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
          return !iface.supportPrefetchTiling(tileDim, _log.nest());
        }
        return true;
    };
    target.markUnknownOpDynamicallyLegal(isUnPrefetchable);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PrefetchTiling>(&ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
} // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPrefetchTilingPass(Logger log) {
    return std::make_unique<PrefetchTilingPass>(log);
}
