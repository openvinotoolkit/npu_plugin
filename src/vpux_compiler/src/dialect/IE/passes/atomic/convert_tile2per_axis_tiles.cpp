//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
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
    explicit ConvertTile2PerAxisTilePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    void runOnFunction() final;

public:
    class TileOpConverter;

private:
    void passBody();

private:
    Logger _log;
};

void ConvertTile2PerAxisTilePass::runOnFunction() {
    try {
        passBody();
    } catch (const std::exception& e) {
        (void)errorAt(getOperation(), "{0} Pass failed : {1}", getName(), e.what());
        signalPassFailure();
    }
}

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

    auto repeats_tensor = origOp.repeats().getDefiningOp<ConstantInterface>();

    if (repeats_tensor == nullptr) {
        return errorAt(origOp->getLoc(), "Got non constant repeats parameters");
    }
    auto repeats = repeats_tensor.getContent().getValues<int64_t>();

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
// passBody
//

void ConvertTile2PerAxisTilePass::passBody() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TileOp>();
    target.addLegalOp<IE::PerAxisTileOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TileOpConverter>(&ctx, _log.nest());

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertShapeTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertTile2PerAxisTilePass(Logger log) {
    return std::make_unique<ConvertTile2PerAxisTilePass>(log);
}
