//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

//
// MemPermuteRewriter
//

class MemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
        this->setDebugName("MemPermuteRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    vpux::NDTypeInterface composeType(vpux::NDTypeInterface origType, const DimsOrder inputOrder,
                                      mlir::AffineMap memPerm) const;
    Logger _log;
};

vpux::NDTypeInterface MemPermuteRewriter::composeType(vpux::NDTypeInterface origType, const DimsOrder inputOrder,
                                                      mlir::AffineMap memPerm) const {
    const auto targetOrder = vpux::applyPermutation(inputOrder, DimsOrder::fromAffineMap(memPerm));
    auto adjustOrder = vpux::moveD0ToTheFront(targetOrder);
    return origType.changeDimsOrder(adjustOrder);
}

mlir::LogicalResult MemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }

    auto layerWithPermute = getFusableLayerWithPermuteInterface(origOp.getOperation());
    if (layerWithPermute == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteRewriter applies for NCE tasks");
    }

    if (!layerWithPermute.isSupportedPermutation(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation does not support {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }

    if (!layerWithPermute->getResult(0).hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "ReorderRewriter applies only for NCE tasks with one consumer");
    }

    auto output = layerWithPermute->getResult(0);
    const auto origType = output.getType().cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE task does not implement vpux::NDTypeInterface");
    }

    auto maybeQuantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*(layerWithPermute->getUsers().begin()));

    const auto newType = composeType(origType, inOrder, origOp.getMemPerm());
    layerWithPermute->getResult(0).setType(newType);

    auto ctx = rewriter.getContext();
    const auto orderInAttr = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.getOutput()).toAffineMap(ctx));
    auto outLayoutCast =
            rewriter.createOrFold<IE::LayoutCastOp>(origOp.getLoc(), layerWithPermute->getResult(0), orderInAttr);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), layerWithPermute->getLoc());

    const auto targetShape = getShape(origOp.getOutput()).raw();
    auto newOutput =
            rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), outLayoutCast, getIntArrayAttr(ctx, targetShape));

    if (maybeQuantizeCastOp != nullptr) {
        newOutput = rewriter.createOrFold<IE::QuantizeCastOp>(
                maybeQuantizeCastOp->getLoc(), origOp.getType(), newOutput,
                maybeQuantizeCastOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType());
    }

    rewriter.replaceOp(origOp, newOutput);

    return mlir::success();
}

//
// SwapSliceMemPermuteRewriter
//

class SwapSliceMemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    SwapSliceMemPermuteRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
        this->setDebugName("SwapSliceMemPermuteRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapSliceMemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(getShape(origOp.getInput()));
    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }

    if (getShape(origOp.getInput()) != getShape(origOp.getOutput())) {
        return mlir::failure();
    }

    auto preSliceOp = origOp.getInput().getDefiningOp<IE::SliceOp>();
    if (preSliceOp == nullptr || !preSliceOp->hasOneUse()) {
        return mlir::failure();
    }

    auto preConvOp = preSliceOp.getOperand().getDefiningOp<IE::ConvolutionOp>();
    if (preConvOp == nullptr || !preConvOp->hasOneUse()) {
        return mlir::failure();
    }

    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(origOp->getLoc(), preConvOp.getOutput(),
                                                             origOp.getDstOrderAttr(), origOp.getMemPermAttr());

    auto layerWithPermute = getFusableLayerWithPermuteInterface(newMemPermuteOp.getOperation());
    if (layerWithPermute == nullptr) {
        return mlir::failure();
    }

    if (!layerWithPermute.isSupportedPermutation(newMemPermuteOp)) {
        return mlir::failure();
    }

    auto newSliceOp = rewriter.create<IE::SliceOp>(origOp->getLoc(), newMemPermuteOp.getOutput(),
                                                   preSliceOp.getStaticOffsetsAttr(), preSliceOp.getStaticSizesAttr());
    rewriter.replaceOp(origOp, newSliceOp.getOutputs());

    return mlir::success();
}

//
// FuseMemPermutePass
//

class FuseMemPermutePass final : public IE::FuseMemPermutePassBase<FuseMemPermutePass> {
public:
    explicit FuseMemPermutePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FuseMemPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapSliceMemPermuteRewriter>(&ctx, benefitLevels[0], _log);
    patterns.add<MemPermuteRewriter>(&ctx, benefitLevels[1], _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMemPermutePass(Logger log) {
    return std::make_unique<FuseMemPermutePass>(log);
}
