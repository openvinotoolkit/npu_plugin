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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertTile2PerAxisTile
//

class ConvertTile2PerAxisTilePass final : public IE::ConvertTile2PerAxisTileBase<ConvertTile2PerAxisTilePass> {
public:
    explicit ConvertTile2PerAxisTilePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class TileOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// GenericOpConverter
//

class ConvertTile2PerAxisTilePass::TileOpConverter final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    TileOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertTile2PerAxisTilePass::TileOpConverter::matchAndRewrite(
        IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Tile Operation '{0}'", origOp->getLoc());

    auto repeatsConst = origOp.repeats().getDefiningOp<Const::DeclareOp>();
    if (repeatsConst == nullptr) {
        return errorAt(origOp->getLoc(), "Got non constant repeats parameters");
    }

    const auto repeatsContent = repeatsConst.content();
    const auto repeats = repeatsContent.getValues<int64_t>();

    const auto gapSize = static_cast<int>(getShape(origOp.input()).size()) - static_cast<int>(repeats.size());

    if (gapSize < 0) {
        _log.error("Op: {0}. Rank of input tensor less then input repeats array size. This case should be handled by "
                   "canonicalizer of TileOp",
                   origOp->getLoc());
        return mlir::failure();
    }

    mlir::Value lastResult = origOp.input();
    for (size_t i = 0; i < repeats.size(); ++i) {
        if (repeats[i] > 1) {
            lastResult = rewriter.create<IE::PerAxisTileOp>(origOp->getLoc(), lastResult, i + gapSize, repeats[i])
                                 .getResult();
        }
    }
    if (lastResult == origOp.input()) {
        return mlir::failure();
    }
    rewriter.replaceOp(origOp, lastResult);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertTile2PerAxisTilePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TileOp>();
    target.addLegalOp<IE::PerAxisTileOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TileOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertTile2PerAxisTilePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertTile2PerAxisTilePass(Logger log) {
    return std::make_unique<ConvertTile2PerAxisTilePass>(log);
}
