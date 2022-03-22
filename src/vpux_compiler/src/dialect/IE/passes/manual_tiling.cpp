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
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// ManualTiling
//

class ManualTiling final : public mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface> {
public:
    ManualTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("ManualTiling");
    }
    mlir::LogicalResult matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ManualTiling::matchAndRewrite(IE::TilingBuilderOpInterface origOp,
                                                  mlir::PatternRewriter& rewriter) const {
    // Manual tiling strategy use the specified number of tiles
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::dyn_cast<IE::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    const auto outputShape = getShape(op->getResult(0));
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Unsupported operation '{0}' at '{1}', it has non 4D result",
                      op->getName(), op->getLoc());

    auto manualTiling = Shape(parseIntArrayAttr<int64_t>(op->getAttr("manualTilingStrategy").cast<mlir::ArrayAttr>()));
    _log.nest(1).trace("Using manual tiles for op {0} at {1}, tiles: {2}", op->getName(), op->getLoc(), manualTiling);
    const auto tiles = vpux::fillDividedTiles(manualTiling, outputShape);

    op->setAttr("manualTilingStrategyApplied", mlir::BoolAttr::get(op->getContext(), true));
    op->setAttr("tilingStrategy", op->getAttr("manualTilingStrategy"));
    op->removeAttr("manualTilingStrategy");

    _log.nest(1).trace("Create {0} tiles:", tiles.size());
    return applyTileStrategy(origOp, tiles, rewriter, _log);
}

//
// ManualTilingPass
//
class ManualTilingPass final : public IE::ManualTilingBase<ManualTilingPass> {
public:
    explicit ManualTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void ManualTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp, IE::ConcatOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp>();
    target.markOpRecursivelyLegal<VPU::NCEClusterTilingOp>([&](mlir::Operation*) {
        return true;
    });
    target.markUnknownOpDynamicallyLegal([this](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::TilingInfoOpInterface>(op)) {
            if (op->hasAttr("manualTilingStrategy") && !op->hasAttr("manualTilingStrategyApplied")) {
                // manual strategy overwrite
                return false;
            }
        }

        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ManualTiling>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createManualTilingPass(Logger log) {
    return std::make_unique<ManualTilingPass>(log);
}
