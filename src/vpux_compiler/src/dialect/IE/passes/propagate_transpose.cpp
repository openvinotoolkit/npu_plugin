//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// MoveThroughSoftmax
//

class MoveThroughSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    MoveThroughSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto transposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto softmaxInputRank = origOp.getInput().getType().dyn_cast<NDTypeInterface>().getRank();
    const auto softmaxAxisInd = getPositiveAxisInd(origOp.getAxisIndAttr(), softmaxInputRank);

    const auto transposePerm = DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    const auto newSoftmaxAxisInd = transposePerm.dimAt(softmaxAxisInd).ind();

    auto newSoftmaxOp =
            rewriter.create<IE::SoftMaxOp>(origOp.getLoc(), transposeOp.getInput().getType(), transposeOp.getInput(),
                                           getIntAttr(getContext(), newSoftmaxAxisInd), origOp.getPadSizeAttr());
    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newSoftmaxOp.getOutput(),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    origOp.replaceAllUsesWith(newTransposeOp.getOutput());

    return mlir::success();
}

//
// MoveThroughGelu
//

class MoveThroughGelu final : public mlir::OpRewritePattern<IE::GeluOp> {
public:
    MoveThroughGelu(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GeluOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughGelu");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::GeluOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveThroughGelu::matchAndRewrite(IE::GeluOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto transposeOp = origOp.getInput().getDefiningOp<IE::TransposeOp>();
    if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "TransposeOp not found or has multiple uses");
    }

    const auto transposeOrder = transposeOp.getOrderValue();
    if (!transposeOrder.has_value()) {
        return matchFailed(_log, rewriter, origOp, "Found invalid TransposeOp");
    }

    auto newGelu =
            rewriter.create<IE::GeluOp>(origOp.getLoc(), transposeOp.getInput().getType(), transposeOp.getInput());
    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), newGelu.getOutput(),
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    origOp.replaceAllUsesWith(newTransposeOp.getOutput());

    return mlir::success();
}

//
// PropagateTransposePass
//

class PropagateTransposePass final : public IE::PropagateTransposeBase<PropagateTransposePass> {
public:
    explicit PropagateTransposePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateTransposePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughSoftmax>(&ctx, _log);
    patterns.add<MoveThroughGelu>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateTransposePass(Logger log) {
    return std::make_unique<PropagateTransposePass>(log);
}
