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

struct MemPermuteConfig {
    MemPermuteConfig(const DimsOrder inputOrder, const DimsOrder dstOrder, const DimsOrder memPerm)
            : _inputOrder(inputOrder), _dstOrder(dstOrder), _memPerm(memPerm) {
    }
    const DimsOrder _inputOrder;
    const DimsOrder _dstOrder;
    const DimsOrder _memPerm;
};

struct MaxPoolConfig {
    MaxPoolConfig(const DimsOrder outputLayout, const DimsOrder dimsOrder)
            : _outputLayout(outputLayout), _dimsOrder(dimsOrder) {
    }
    const DimsOrder _outputLayout;
    const DimsOrder _dimsOrder;
};

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
    Logger _log;
};

mlir::LogicalResult MemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    const auto inShape = getShape(origOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (isTrivialPermute(inMemShape, origOp.mem_perm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }
    const auto dstOrder = DimsOrder::fromAffineMap(origOp.dst_order());
    const auto memPerm = DimsOrder::fromAffineMap(origOp.mem_perm());

    const SmallVector<std::pair<MemPermuteConfig, MaxPoolConfig>> configMapping = {
            {MemPermuteConfig(DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC),
             MaxPoolConfig(DimsOrder::NCWH, DimsOrder::NHCW)},
            {MemPermuteConfig(DimsOrder::NHCW, DimsOrder::NCHW, DimsOrder::NHCW),
             MaxPoolConfig(DimsOrder::NWHC, DimsOrder::NWHC)},
            {MemPermuteConfig(DimsOrder::NCHW, DimsOrder::NCHW, DimsOrder::NHWC),
             MaxPoolConfig(DimsOrder::NWCH, DimsOrder::NWCH)},
            {MemPermuteConfig(DimsOrder::NCHW, DimsOrder::NCHW, DimsOrder::NHCW),
             MaxPoolConfig(DimsOrder::NWHC, DimsOrder::NWCH)},
    };

    const auto configMatcher = [&](const std::pair<MemPermuteConfig, MaxPoolConfig>& conf) -> bool {
        const auto& memPermuteConfig = conf.first;
        return memPermuteConfig._inputOrder == inOrder && memPermuteConfig._dstOrder == dstOrder &&
               memPermuteConfig._memPerm == memPerm;
    };
    const auto configIter = std::find_if(configMapping.begin(), configMapping.end(), configMatcher);
    if (configIter == configMapping.end()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Unsupported configuration.");
    }

    const auto targetOrder = configIter->second._outputLayout;
    const auto dimsOrder = configIter->second._dimsOrder.toPermutation();
    const auto targetInShape = {inShape[dimsOrder[0]], inShape[dimsOrder[1]], inShape[dimsOrder[2]],
                                inShape[dimsOrder[3]]};

    auto ctx = rewriter.getContext();
    auto reshapedInput = rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), origOp->getOperand(0),
                                                          getIntArrayAttr(ctx, targetInShape));
    const auto inOrderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto inLayoutCast = rewriter.create<IE::LayoutCastOp>(origOp.getLoc(), reshapedInput.result(), inOrderAttr);

    const auto layoutCastType = inLayoutCast.output().getType().cast<NDTypeInterface>();
    const auto outType = layoutCastType.changeDimsOrder(targetOrder);
    auto maxPool = createIdentityMaxPool(inLayoutCast.output(), outType, rewriter);
    auto alignInterface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(maxPool);
    if (alignInterface == nullptr || alignInterface.verifyChannels().failed()) {
        rewriter.eraseOp(maxPool);
        rewriter.eraseOp(inLayoutCast);
        rewriter.eraseOp(reshapedInput);
        return matchFailed(_log.nest(), rewriter, origOp, "Channels of an IE.MaxPool are not aligned.");
    }

    const auto orderInAttr = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.output()).toAffineMap(ctx));
    auto outLayoutCast = rewriter.createOrFold<IE::LayoutCastOp>(origOp.getLoc(), maxPool->getResult(0), orderInAttr);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), maxPool->getLoc());

    const auto targetShape = getShape(origOp.output()).raw();
    auto reshapedOut = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), origOp.getType(), outLayoutCast,
                                                              getIntArrayAttr(ctx, targetShape));
    rewriter.replaceOp(origOp, reshapedOut);

    return mlir::success();
}

//
// ConvertMemPermuteToPoolPass
//

class ConvertMemPermuteToPoolPass final : public IE::ConvertMemPermuteToPoolPassBase<ConvertMemPermuteToPoolPass> {
public:
    explicit ConvertMemPermuteToPoolPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertMemPermuteToPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemPermuteRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> IE::createConvertMemPermuteToPoolPass(Logger log) {
    return std::make_unique<ConvertMemPermuteToPoolPass>(log);
}
