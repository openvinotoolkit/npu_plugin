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

template <class ConcreteOp>
SmallVector<mlir::Type> inferOutputTypes(ConcreteOp origOp, mlir::ValueRange newInputs, ArrayRef<DimsOrder> newOrders) {
    SmallVector<mlir::Type> newTypes;
    VPUX_THROW_UNLESS(ConcreteOp::inferReturnTypes(origOp->getContext(), origOp->getLoc(), newInputs,
                                                   origOp->getAttrDictionary(), origOp->getRegions(), newTypes)
                              .succeeded(),
                      "Failed to infer new output type for operation '{0}' at '{1}'", origOp->getName(),
                      origOp->getLoc());

    VPUX_THROW_UNLESS(newOrders.size() == newTypes.size(),
                      "New orders size '{0}' doesn't match new output types size '{1}'", newOrders.size(),
                      newTypes.size());

    for (auto i : irange(newTypes.size())) {
        newTypes[i] = changeDimsOrder(newTypes[i].cast<mlir::ShapedType>(), newOrders[i]);
    }

    return newTypes;
}

template <class ConcreteOp>
mlir::Type inferOutputType(ConcreteOp origOp, mlir::ValueRange newInputs, DimsOrder newOrder) {
    const auto newTypes = inferOutputTypes(origOp, newInputs, {newOrder});
    VPUX_THROW_UNLESS(newTypes.size() == 1, "Got wrong number of inferred types - '{0}'", newTypes.size());
    return newTypes[0];
}

//
// ReorderWithSubView
//

class ReorderWithSubView final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ReorderWithSubView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ReorderWithSubView");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origSubViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSubView::matchAndRewrite(IE::SliceOp origSubViewOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origSubViewOp.source().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> subview at '{1}' pair", origReorderOp->getLoc(), origSubViewOp->getLoc());

    if (!origReorderOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origSubViewOp, "Reorder has more then one user");
    }

    const auto newSubViewType =
            inferOutputType(origSubViewOp, origReorderOp.input(), DimsOrder::fromValue(origReorderOp.input()));
    auto newSubViewOp = rewriter.create<IE::SliceOp>(origSubViewOp->getLoc(), newSubViewType, origReorderOp.input(),
                                                     origSubViewOp.static_offsets(), origSubViewOp.static_sizes());

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

void swapExpandWithReorder(mlir::PatternRewriter& rewriter, IE::ExpandOp expandOp, IE::ReorderOp origReorderOp) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(expandOp);

    const auto newExpandType =
            inferOutputType(expandOp, origReorderOp.input(), DimsOrder::fromValue(origReorderOp.input()));
    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp->getLoc(), newExpandType, origReorderOp.input(),
                                                     expandOp.pads_begin(), expandOp.pads_end());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(expandOp, newExpandOp.output(), origReorderOp.dstOrderAttr());
}

mlir::LogicalResult ReorderWithExpand::matchAndRewrite(IE::ExpandOp origExpandOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origExpandOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Expand at '{1}' pair", origReorderOp->getLoc(), origExpandOp->getLoc());
    const auto isExpand = [](const mlir::Operation* reorderUser) -> bool {
        return mlir::isa<IE::ExpandOp>(reorderUser);
    };

    if (!std::all_of(origReorderOp.getResult().user_begin(), origReorderOp.getResult().user_end(), isExpand)) {
        return matchFailed(_log.nest(), rewriter, origExpandOp,
                           "Reorder has more than one user and they are heterogeneous");
    }

    for (const auto reorderUser : origReorderOp.getResult().getUsers()) {
        auto expandOp = mlir::dyn_cast_or_null<IE::ExpandOp>(reorderUser);
        swapExpandWithReorder(rewriter, expandOp, origReorderOp);
    }

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
    }

    SmallVector<DimsOrder> newOrders(origSplitOp->getNumResults(), DimsOrder::fromValue(origReorderOp.input()));
    const auto newOutputTypes = inferOutputTypes(origSplitOp, origReorderOp.input(), newOrders);

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

    const auto newConcatType = inferOutputType(origConcatOp, initialInputs, initialOrder.getValue());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origConcatOp, newConcatType, initialInputs, origConcatOp.axisAttr());

    return mlir::success();
}

//
// ReorderWithLayer
//

class ReorderWithLayer final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    ReorderWithLayer(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _log(log) {
        setDebugName("ReorderWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithLayer::matchAndRewrite(IE::LayoutInfoOpInterface layerOp,
                                                      mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<IE::ReorderOp>(layerOp)) {
        return mlir::failure();
    }

    _log.trace("Got layer operation '{0}' at '{1}'", layerOp->getName(), layerOp->getLoc());

    auto orderInfo = layerOp.getLayoutInfo();
    _log.nest(1).trace("Base order info - '{0}'", orderInfo);

    _log.nest(1).trace("Check operation inputs");
    for (auto ind : irange(orderInfo.getNumInputs())) {
        const auto arg = layerOp->getOperand(checked_cast<uint32_t>(ind));

        auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            continue;
        }

        const auto newOrder = DimsOrder::fromValue(argReorderOp.input());

        _log.nest(2).trace("Try '{0}' order for input #{1}", newOrder, ind);
        orderInfo.setInput(ind, newOrder);
    }

    _log.nest(1).trace("Check operation outputs");
    for (auto ind : irange(orderInfo.getNumOutputs())) {
        const auto res = layerOp->getResult(checked_cast<uint32_t>(ind));

        if (!res.hasOneUse()) {
            continue;
        }

        auto resReorderOp = mlir::dyn_cast<IE::ReorderOp>(*res.user_begin());
        if (resReorderOp == nullptr) {
            continue;
        }

        const auto newOrder = DimsOrder::fromValue(resReorderOp.output());

        _log.nest(2).trace("Try '{0}' order for output #{1}", newOrder, ind);
        orderInfo.setOutput(ind, newOrder);
    }

    if (!orderInfo.hasChanges()) {
        return matchFailed(_log.nest(), rewriter, layerOp, "There is no Reorders to fuse");
    }

    orderInfo.resetChanges();
    layerOp.inferLayoutInfo(orderInfo);

    if (orderInfo.hasChanges()) {
        return matchFailed(_log.nest(), rewriter, layerOp, "The layer doesn't support fused Reorders");
    }

    _log.nest(1).trace("Merge new order - '{0}'", orderInfo);

    rewriter.startRootUpdate(layerOp);

    for (auto ind : irange(orderInfo.getNumInputs())) {
        auto& arg = layerOp->getOpOperand(checked_cast<uint32_t>(ind));

        auto argReorderOp = arg.get().getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            continue;
        }

        arg.set(argReorderOp.input());
    }
    for (auto ind : irange(orderInfo.getNumOutputs())) {
        auto res = layerOp->getResult(checked_cast<uint32_t>(ind));

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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderWithSubView>(&ctx, _log);
    patterns.add<ReorderWithExpand>(&ctx, _log);
    patterns.add<ReorderWithSplit>(&ctx, _log);
    patterns.add<ReorderWithConcat>(&ctx, _log);
    patterns.add<ReorderWithLayer>(&ctx, _log);
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
