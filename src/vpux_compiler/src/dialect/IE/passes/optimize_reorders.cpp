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

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ReorderWithSubView
//

class ReorderWithSubView final : public mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
public:
    ReorderWithSubView(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp>(ctx), _log(log) {
        setDebugName("ReorderWithSubView");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::tensor::ExtractSliceOp origSubViewOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSubView::matchAndRewrite(mlir::tensor::ExtractSliceOp origSubViewOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origSubViewOp.source().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> subview at '{1}' pair", origReorderOp->getLoc(), origSubViewOp->getLoc());

    if (!origReorderOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origSubViewOp, "Reorder has more then one user");
    }

    const auto subViewShape = getShape(origSubViewOp.result());
    const auto newSubViewType = changeShape(origReorderOp.input().getType().cast<mlir::ShapedType>(), subViewShape);
    auto newSubViewOp = rewriter.create<mlir::tensor::ExtractSliceOp>(
            origSubViewOp->getLoc(), newSubViewType, origReorderOp.input(), origSubViewOp.offsets(),
            origSubViewOp.sizes(), origSubViewOp.strides(), origSubViewOp.static_offsets(),
            origSubViewOp.static_sizes(), origSubViewOp.static_strides());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origSubViewOp, newSubViewOp.result(), origReorderOp.dstOrderAttr());
    return mlir::success();
}

//
// ReorderWithExpand
//

class ReorderWithExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ReorderWithExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("ReorderWithExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithExpand::matchAndRewrite(IE::ExpandOp origExpandOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origExpandOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Expand at '{1}' pair", origReorderOp->getLoc(), origExpandOp->getLoc());

    if (!origReorderOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origExpandOp, "Reorder has more then one users");
    }

    const auto expandShape = getShape(origExpandOp.output());
    const auto newExpandType = changeShape(origReorderOp.input().getType().cast<mlir::ShapedType>(), expandShape);
    auto newExpandOp = rewriter.create<IE::ExpandOp>(origExpandOp->getLoc(), newExpandType, origReorderOp.input(),
                                                     origExpandOp.pads_begin(), origExpandOp.pads_end());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origExpandOp, newExpandOp.output(), origReorderOp.dstOrderAttr());
    return mlir::success();
}

//
// ReorderWithSplit
//

class ReorderWithSplit final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    ReorderWithSplit(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("ReorderWithSplit");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSplit::matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const {
    if (origSplitOp.axis() != nullptr) {
        return mlir::failure();
    }

    auto origReorderOp = origSplitOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Split at '{1}' pair", origReorderOp->getLoc(), origSplitOp->getLoc());

    const auto initialOrder = DimsOrder::fromValue(origReorderOp.input());

    SmallVector<IE::ReorderOp> outputReorders;
    outputReorders.reserve(origSplitOp.outputs().size());

    SmallVector<mlir::Type> newOutputTypes;
    newOutputTypes.reserve(origSplitOp.outputs().size());

    for (auto res : origSplitOp.outputs()) {
        if (!res.hasOneUse()) {
            return matchFailed(_log.nest(), rewriter, origSplitOp, "Split output #{0} has more then one user",
                               res.getResultNumber());
        }

        auto resReorderOp = mlir::dyn_cast<IE::ReorderOp>(*res.user_begin());
        if (resReorderOp == nullptr) {
            return matchFailed(_log.nest(), rewriter, origSplitOp, "Split output #{0} consumed by non Reorder",
                               res.getResultNumber());
        }

        const auto resOrder = DimsOrder::fromValue(resReorderOp.output());
        if (resOrder != initialOrder) {
            return matchFailed(_log.nest(), rewriter, origSplitOp, "Split output #{0} is reordered to different order",
                               res.getResultNumber());
        }

        outputReorders.push_back(resReorderOp);

        const auto newResType = changeDimsOrder(res.getType().cast<mlir::ShapedType>(), initialOrder);
        newOutputTypes.push_back(newResType);
    }

    auto newSplitOp = rewriter.create<IE::SplitOp>(origSplitOp->getLoc(), newOutputTypes, origReorderOp.input(),
                                                   origSplitOp.axis(), origSplitOp.num_splitsAttr(),
                                                   origSplitOp.axis_valueAttr());

    for (auto ind : irange(outputReorders.size())) {
        auto oldResReorderOp = outputReorders[ind];
        auto newResVal = newSplitOp->getResult(checked_cast<uint32_t>(ind));
        rewriter.replaceOp(oldResReorderOp, newResVal);
    }

    return mlir::success();
}

//
// ReorderWithConcat
//

class ReorderWithConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ReorderWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ReorderWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConcat::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                       mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> initialInputs;
    initialInputs.reserve(origConcatOp.inputs().size());

    Optional<DimsOrder> initialOrder;

    for (auto arg : origConcatOp.inputs()) {
        auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            return mlir::failure();
        }

        const auto argOrder = DimsOrder::fromValue(argReorderOp.input());
        if (!initialOrder.hasValue()) {
            initialOrder = argOrder;
        } else if (argOrder != initialOrder.getValue()) {
            return mlir::failure();
        }

        initialInputs.push_back(argReorderOp.input());
    }

    if (!initialOrder.hasValue()) {
        return mlir::failure();
    }

    if (!origConcatOp.output().hasOneUse()) {
        return mlir::failure();
    }

    auto resReorderOp = mlir::dyn_cast<IE::ReorderOp>(*origConcatOp.output().user_begin());
    if (resReorderOp == nullptr) {
        return mlir::failure();
    }

    const auto resOrder = DimsOrder::fromValue(resReorderOp.output());
    if (resOrder != initialOrder.getValue()) {
        return mlir::failure();
    }

    const auto newType = changeDimsOrder(origConcatOp.getType(), initialOrder.getValue());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origConcatOp, newType, initialInputs, origConcatOp.axisAttr());

    return mlir::success();
}

#if 0

// FIXME: isSupportedLayout should use orderInfo to get layouts for inputs and outputs,
//        not take them from the operands.

//
// ReorderWithLayer
//

class ReorderWithLayer final : public mlir::OpInterfaceRewritePattern<IE::LayerOpInterface> {
public:
    ReorderWithLayer(mlir::MLIRContext* ctx, const IE::LayerInfoDialectInterface* layerInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerOpInterface>(ctx), _layerInfo(layerInfo), _log(log) {
        setDebugName("ReorderWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    const IE::LayerInfoDialectInterface* _layerInfo = nullptr;
    Logger _log;
};

mlir::LogicalResult ReorderWithLayer::matchAndRewrite(IE::LayerOpInterface layerOp, mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<mlir::ViewLikeOpInterface, IE::SplitOp, IE::ConcatOp, IE::ExpandOp, IE::ReorderOp>(
                layerOp.getOperation())) {
        return mlir::failure();
    }

    auto orderInfo = layerOp.getDataOrderInfo();

    bool hasChanges = false;
    for (const auto p : layerOp.getInputs() | indexed) {
        const auto arg = p.value();

        auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            continue;
        }

        orderInfo.setInput(p.index(), DimsOrder::fromValue(argReorderOp.input()));
        hasChanges = true;
    }
    for (const auto p : layerOp.getOutputs() | indexed) {
        const auto res = p.value();

        if (!res.hasOneUse()) {
            continue;
        }

        auto resReorderOp = mlir::dyn_cast<IE::ReorderOp>(*res.user_begin());
        if (resReorderOp == nullptr) {
            continue;
        }

        orderInfo.setOutput(p.index(), DimsOrder::fromValue(resReorderOp.output()));
        hasChanges = true;
    }

    if (!hasChanges) {
        return mlir::failure();
    }

    if (!_layerInfo->isSupportedLayout(layerOp, orderInfo)) {
        return mlir::failure();
    }

    rewriter.startRootUpdate(layerOp);

    for (const auto p : layerOp->getOpOperands() | indexed) {
        auto& arg = p.value();

        auto argReorderOp = arg.get().getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            continue;
        }

        arg.set(argReorderOp.input());
    }
    for (const auto p : layerOp->getOpResults() | indexed) {
        auto res = p.value();

        if (!res.hasOneUse()) {
            continue;
        }

        auto resReorderOp = mlir::dyn_cast<IE::ReorderOp>(*res.user_begin());
        if (resReorderOp == nullptr) {
            continue;
        }

        const auto origType = res.getType().cast<mlir::ShapedType>();
        const auto newType = changeDimsOrder(origType, DimsOrder::fromValue(resReorderOp.output()));
        res.setType(newType);
    }

    rewriter.finalizeRootUpdate(layerOp);

    return mlir::success();
}

#endif

//
// OptimizeReordersPass
//

class OptimizeReordersPass final : public IE::OptimizeReordersBase<OptimizeReordersPass> {
public:
    explicit OptimizeReordersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeReordersPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto* dialect = ctx.getOrLoadDialect<IE::IEDialect>();
    VPUX_THROW_UNLESS(dialect != nullptr, "IE Dialect was not loaded");

    const auto* layerInfo = dialect->getRegisteredInterface<IE::LayerInfoDialectInterface>();
    VPUX_THROW_UNLESS(layerInfo != nullptr, "LayerInfoDialect is not registered");

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderWithSubView>(&ctx, _log);
    patterns.add<ReorderWithExpand>(&ctx, _log);
    patterns.add<ReorderWithSplit>(&ctx, _log);
    patterns.add<ReorderWithConcat>(&ctx, _log);
    // patterns.add<ReorderWithLayer>(&ctx, layerInfo, _log);
    IE::ReorderOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeReordersPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeReordersPass(Logger log) {
    return std::make_unique<OptimizeReordersPass>(log);
}
