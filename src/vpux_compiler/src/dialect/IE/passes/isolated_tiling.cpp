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

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/FunctionExtras.h>

#include <numeric>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

OutputTiling generateTiles(IE::TilingBuilderOpInterface origOp, Logger log) {
    auto* op = origOp.getOperation();
    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    // Manual Strategy Utils
    if (op->hasAttr("tilingStrategy")) {
        // if manual tiling strategy use the specified number of tiles
        auto manualTiling = Shape(parseIntArrayAttr<int64_t>(op->getAttr("tilingStrategy").cast<mlir::ArrayAttr>()));
        log.trace("Using manual tiles for op {0} at {1}, tiles: {2}", op->getName(), op->getLoc(), manualTiling);
        return vpux::fillDividedTiles(manualTiling, outputShape);
    }

    // create tiles for operation
    auto nTilesOnDim = IE::computeGeneralTileStrategy(op, log);

    // store tiles for operations
    const auto tilesAttr = getIntArrayAttr(op->getContext(), nTilesOnDim);
    op->setAttr("tilingStrategy", tilesAttr);

    return vpux::fillDividedTiles(nTilesOnDim, outputShape);
}

//
// GenericTiling
//

class GenericTiling final : public mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface> {
public:
    GenericTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("GenericTiling");
    }

    mlir::LogicalResult matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GenericTiling::matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto tiles = generateTiles(origOp, _log);
    _log.nest(1).trace("Create {0} tiles:", tiles.size());

    return applyTileStrategy(origOp, tiles, rewriter, _log);
}

//
// IsolatedTilingPass
//

class IsolatedTilingPass final : public IE::IsolatedTilingBase<IsolatedTilingPass> {
public:
    explicit IsolatedTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void IsolatedTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp, IE::ConcatOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp>();
    target.markOpRecursivelyLegal<VPU::NCEClusterTilingOp>([&](mlir::Operation*) {
        return true;
    });
    target.markUnknownOpDynamicallyLegal([this](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
            const auto resShape = getShape(op->getResult(0));
            return iface.isSupportedTiling({TileInfo(resShape)}, _log.nest());
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

std::unique_ptr<mlir::Pass> vpux::IE::createIsolatedTilingPass(Logger log) {
    return std::make_unique<IsolatedTilingPass>(log);
}
