//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

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
    MemPermuteRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
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

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    if (inShape[Dim(0)] != 1) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp with dim N > 1");
    }

    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }
    const auto dstOrder = DimsOrder::fromAffineMap(origOp.getDstOrder());
    const auto memPerm = DimsOrder::fromAffineMap(origOp.getMemPerm());

    const SmallVector<std::pair<MemPermuteConfig, MaxPoolConfig>> configMapping = {
            {MemPermuteConfig(DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC),
             MaxPoolConfig(DimsOrder::NCWH, DimsOrder::NHCW)},
            {MemPermuteConfig(DimsOrder::NHCW, DimsOrder::NCHW, DimsOrder::NHCW),
             MaxPoolConfig(DimsOrder::NWHC, DimsOrder::NWHC)},
            {MemPermuteConfig(DimsOrder::NCHW, DimsOrder::NCHW, DimsOrder::NHWC),
             MaxPoolConfig(DimsOrder::NWCH, DimsOrder::NWCH)},
            {MemPermuteConfig(DimsOrder::NCHW, DimsOrder::NCHW, DimsOrder::NHCW),
             MaxPoolConfig(DimsOrder::NWHC, DimsOrder::NWCH)},
            {MemPermuteConfig(DimsOrder::NCHW, DimsOrder::NCHW, DimsOrder::NCWH),
             MaxPoolConfig(DimsOrder::NHCW, DimsOrder::NWCH)},
            {MemPermuteConfig(DimsOrder::NHWC, DimsOrder::NCHW, DimsOrder::NWCH),
             MaxPoolConfig(DimsOrder::NCHW, DimsOrder::NCHW)},
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
    auto inLayoutCast = rewriter.create<IE::LayoutCastOp>(origOp.getLoc(), reshapedInput.getResult(), inOrderAttr);

    const auto layoutCastType = inLayoutCast.getOutput().getType().cast<NDTypeInterface>();
    const auto outType = layoutCastType.changeDimsOrder(targetOrder);
    auto maxPool = IE::createIdentityMaxPool(inLayoutCast.getOutput(), outType, rewriter);
    auto alignInterface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(maxPool);
    if (alignInterface == nullptr || alignInterface.verifyChannels().failed()) {
        rewriter.eraseOp(maxPool);
        rewriter.eraseOp(inLayoutCast);
        rewriter.eraseOp(reshapedInput);
        return matchFailed(_log.nest(), rewriter, origOp, "Channels of an IE.MaxPool are not aligned.");
    }

    const auto orderInAttr = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.getOutput()).toAffineMap(ctx));
    auto outLayoutCast = rewriter.createOrFold<IE::LayoutCastOp>(origOp.getLoc(), maxPool->getResult(0), orderInAttr);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), maxPool->getLoc());

    const auto targetShape = getShape(origOp.getOutput()).raw();
    auto reshapedOut = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), origOp.getType(), outLayoutCast,
                                                              getIntArrayAttr(ctx, targetShape));
    rewriter.replaceOp(origOp, reshapedOut);

    return mlir::success();
}

//
// AdjustMemPermuteShapeRewriter
//
class AdjustMemPermuteShapeRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    AdjustMemPermuteShapeRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
        this->setDebugName("AdjustMemPermuteShapeRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// To handle MemPermuteOP's input channel not align from NHWC to NCHW
//   Consider this case:
//       N   H   W   C     ->  N   C   H   W
//       1  inH inW inC        1  inC inH inW
//   The input Channel is not aligned, we can use below conversion to avoid the PermuteDMA
//
// N   H   W   C
// 1  inH inW inC
//       | ShapeCast
//       V
// N   H     W    C
// 1  inH inWxinC 1
//       | LayoutCast
//       V
// N  C   H     W
// 1  1  inH inWxinC
//       | MemPermute
//       V
// N  C     H     W
// 1  1  inWxinC inH
//       | ShapeCast
//       V
// N  C   H     W
// 1  1  inW inCxinH
//       | MemPermute
//       V
// N  C     H     W
// 1  1  inCxinH inW
//       | ShapeCast
//       V
// N   C   H   W
// 1  inC inH inW
//
mlir::LogicalResult AdjustMemPermuteShapeRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }
    const auto dstOrder = DimsOrder::fromAffineMap(origOp.getDstOrder());
    const auto memPerm = DimsOrder::fromAffineMap(origOp.getMemPerm());
    if (inOrder != DimsOrder::NHWC || dstOrder != DimsOrder::NCHW || memPerm != DimsOrder::NWCH) {
        return matchFailed(_log.nest(), rewriter, origOp, "Unsupported input {0} and output {1}.", inOrder, dstOrder);
    }

    auto maxPool = IE::createIdentityMaxPool(origOp.getInput(), origOp.getInput().getType(), rewriter);
    auto alignInterface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(maxPool);
    const auto alignedChannel = alignInterface.getInputChannelAlignment();
    // FIXME:#90478 -- don't create temporary operation
    // The new MaxPool only used to get the channel alignment number. So delete it immediately.
    rewriter.eraseOp(maxPool);

    if (inShape[Dims4D::Act::C] % alignedChannel == 0) {
        return matchFailed(_log.nest(), rewriter, origOp, "Channel already aligned");
    }

    const auto wcSize = inShape[Dims4D::Act::C] * inShape[Dims4D::Act::W];
    const auto hcSize = inShape[Dims4D::Act::C] * inShape[Dims4D::Act::H];
    if (wcSize % alignedChannel || hcSize % alignedChannel) {
        return matchFailed(_log.nest(), rewriter, origOp, "Doesn't satisfy NCE's channel alignment");
    }

    Shape newInShape = inShape.toValues();
    newInShape[Dims4D::Act::C] = 1;
    newInShape[Dims4D::Act::W] = wcSize;

    auto ctx = rewriter.getContext();
    auto reshapedInput = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), origOp.getInput(),
                                                                getIntArrayAttr(ctx, newInShape.raw()));

    const auto inOrderAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    auto inLayoutCast = rewriter.create<IE::LayoutCastOp>(origOp.getLoc(), reshapedInput, inOrderAttr);
    auto newMemPermuteNWCH = rewriter.create<IE::MemPermuteOp>(
            origOp.getLoc(), inLayoutCast.getResult(), mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx)),
            mlir::AffineMapAttr::get(DimsOrder::NCWH.toAffineMap(ctx)));

    Shape secondPermuteInShape = inShape.toValues();
    secondPermuteInShape[Dims4D::Act::C] = 1;
    secondPermuteInShape[Dims4D::Act::W] = hcSize;
    secondPermuteInShape[Dims4D::Act::H] = inShape[Dims4D::Act::W];
    auto midReshapedInput = rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), newMemPermuteNWCH.getResult(),
                                                             getIntArrayAttr(ctx, secondPermuteInShape.raw()));
    auto secondMemPermuteNWCH = rewriter.create<IE::MemPermuteOp>(
            origOp.getLoc(), midReshapedInput.getResult(), mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx)),
            mlir::AffineMapAttr::get(DimsOrder::NCWH.toAffineMap(ctx)));

    const auto targetShape = getShape(origOp.getOutput()).raw();
    auto reshapedOut = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), secondMemPermuteNWCH.getResult(),
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
    patterns.add<AdjustMemPermuteShapeRewriter>(&ctx, benefitLevels[0], _log);
    patterns.add<MemPermuteRewriter>(&ctx, benefitLevels[1], _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> IE::createConvertMemPermuteToPoolPass(Logger log) {
    return std::make_unique<ConvertMemPermuteToPoolPass>(log);
}
