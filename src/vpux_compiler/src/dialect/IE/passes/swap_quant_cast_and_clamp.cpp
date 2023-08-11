//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ClampOp>(ctx), _log(log) {
        setDebugName("LayerRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op: {0}", origOp);

    auto quantizeCastOp = origOp.input().getDefiningOp<IE::QuantizeCastOp>();
    if (quantizeCastOp == nullptr) {
        return mlir::failure();
    }

    if (!quantizeCastOp->hasOneUse()) {
        _log.trace("QuantizeCast has more than one use.");
        return mlir::failure();
    }

    auto inElemType = quantizeCastOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = quantizeCastOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();

    auto inQType = inElemType.dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outQType = outElemType.dyn_cast<mlir::quant::UniformQuantizedType>();

    if (inQType == nullptr || outQType == nullptr) {
        _log.trace("Only per-tensor quantization is supported. Got:\n input type: {0}\n output type: {1}", inElemType,
                   outElemType);
        return mlir::failure();
    }

    auto inScale = inQType.getScale();
    auto outScale = outQType.getScale();

    VPUX_THROW_WHEN(isDoubleEqual(outScale, 0.0f), "Output scale is zero");

    auto rescale = inScale / outScale;
    auto min = origOp.minAttr().getValueAsDouble() * rescale;
    auto max = origOp.maxAttr().getValueAsDouble() * rescale;
    _log.trace("min: {0}, max: {1}", min, max);

    const auto minAttr = getFPAttr(rewriter, min);
    const auto maxAttr = getFPAttr(rewriter, max);

    auto newClamp = rewriter.create<IE::ClampOp>(origOp->getLoc(), quantizeCastOp.input(), minAttr, maxAttr);
    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, newClamp.output(), quantizeCastOp.dstElemTypeAttr());

    return mlir::success();
}

//
// SwapQuantCastAndClampPass
//

class SwapQuantCastAndClampPass final : public IE::SwapQuantCastAndClampBase<SwapQuantCastAndClampPass> {
public:
    explicit SwapQuantCastAndClampPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapQuantCastAndClampPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewriter>(&ctx, _log.nest());

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapQuantCastAndClampPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapQuantCastAndClampPass(Logger log) {
    return std::make_unique<SwapQuantCastAndClampPass>(log);
}
