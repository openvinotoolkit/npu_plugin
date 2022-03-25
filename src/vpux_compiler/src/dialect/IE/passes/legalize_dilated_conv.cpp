//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// DilatedConvolutionRewriter
//

class DilatedConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    DilatedConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("DilatedConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DilatedConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    auto dilatedFilter = rewriter.create<IE::ExpandDilatedOp>(origOp->getLoc(), origOp.filter(), origOp.dilations());
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            origOp, origOp.input(), dilatedFilter.getResult(), origOp.bias(), origOp.strides(), origOp.pads_begin(),
            origOp.pads_end(), getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})), origOp.post_opAttr());
    return mlir::success();
}

//
// DilatedGroupConvolutionRewriter
//

class DilatedGroupConvolutionRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DilatedGroupConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("DilatedGroupConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DilatedGroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution layer at '{1}'", getDebugName(), origOp->getLoc());

    auto dilatedFilter = rewriter.create<IE::ExpandDilatedOp>(origOp->getLoc(), origOp.filter(), origOp.dilations());
    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, origOp.input(), dilatedFilter.getResult(), origOp.bias(), origOp.strides(), origOp.pads_begin(),
            origOp.pads_end(), getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})), origOp.groupsAttr(),
            origOp.post_opAttr());
    return mlir::success();
}

//
// LegalizeDilatedConvolutionPass
//

class LegalizeDilatedConvolutionPass final : public IE::LegalizeDilatedConvolutionBase<LegalizeDilatedConvolutionPass> {
public:
    explicit LegalizeDilatedConvolutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LegalizeDilatedConvolutionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasSupportedDilations = [](ArrayRef<int64_t> dilations) {
        return dilations[0] == 1 && dilations[1] == 1;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
        return hasSupportedDilations(dilations);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
        return hasSupportedDilations(dilations);
    });
    target.addLegalOp<IE::ExpandDilatedOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DilatedConvolutionRewriter>(&ctx, _log);
    patterns.insert<DilatedGroupConvolutionRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLegalizeDilatedConvolutionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeDilatedConvolutionPass(Logger log) {
    return std::make_unique<LegalizeDilatedConvolutionPass>(log);
}
