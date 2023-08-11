//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// MemPermuteRewriter
//

class MemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
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
    return origType.changeDimsOrder(targetOrder);
}

mlir::LogicalResult MemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check that reorder is not applied to sub-byte element types:
    const auto elemType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const Bit elemSize = vpux::getElemTypeSize(elemType);
    if (elemSize.count() < CHAR_BIT) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation does not apply to sub-byte types. Got {0}",
                           elemType);
    }

    // Check that permutation is supported by ODU
    const auto outOrder = DimsOrder::fromValue(origOp.output());
    const std::array<DimsOrder, 6> supportedOrders = {
            DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHCW, DimsOrder::NHWC, DimsOrder::NWCH, DimsOrder::NWHC,
    };
    const auto isOutOrder = [&](const DimsOrder supported) -> bool {
        return supported == outOrder;
    };
    if (std::none_of(supportedOrders.cbegin(), supportedOrders.cend(), isOutOrder)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation does not support {0}", outOrder);
    }

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto inShape = getShape(origOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (isTrivialPermute(inMemShape, origOp.mem_perm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderOp is actually a permute cast");
    }

    auto maybeNCEOp = origOp.input().getDefiningOp();
    if (maybeNCEOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderOp does not have defining operation");
    }

    // TODO: Remove this as part of E#72884
    if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AndOp>(maybeNCEOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderRewriter applies only for NCE tasks");
    }

    if (VPUIP::NCEInvariant::verifyKernel(maybeNCEOp, _log).failed()) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderRewriter applies for NCE tasks");
    }

    if (!maybeNCEOp->getResult(0).hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "ReorderRewriter applies only for NCE tasks with one consumer");
    }

    auto output = maybeNCEOp->getResult(0);
    const auto origType = output.getType().cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE task does not implement vpux::NDTypeInterface");
    }

    const auto newType = composeType(origType, inOrder, origOp.mem_perm());
    maybeNCEOp->getResult(0).setType(newType);

    auto ctx = rewriter.getContext();
    const auto orderInAttr = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.output()).toAffineMap(ctx));
    auto outLayoutCast =
            rewriter.createOrFold<IE::LayoutCastOp>(origOp.getLoc(), maybeNCEOp->getResult(0), orderInAttr);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), maybeNCEOp->getLoc());

    const auto targetShape = getShape(origOp.output()).raw();
    auto reshapedOut = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), origOp.getType(), outLayoutCast,
                                                              getIntArrayAttr(ctx, targetShape));
    rewriter.replaceOp(origOp, reshapedOut);

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
    patterns.add<MemPermuteRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseMemPermutePass(Logger log) {
    return std::make_unique<FuseMemPermutePass>(log);
}
