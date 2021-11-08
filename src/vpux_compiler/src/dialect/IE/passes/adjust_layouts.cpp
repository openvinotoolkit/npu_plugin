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
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

using InputToReordersMap = mlir::DenseMap<mlir::Value, std::unordered_map<DimsOrder, IE::ReorderOp>>;

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, InputToReordersMap* const reorders, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _reorders(reorders), _log(log) {
        VPUX_THROW_UNLESS(reorders != nullptr, "Got NULL reorders map");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    IE::ReorderOp createReorder(mlir::Operation* op, mlir::Value input, DimsOrder dstOrder,
                                mlir::PatternRewriter& rewriter) const;

    void insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                               mlir::PatternRewriter& rewriter) const;
    void insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                mlir::PatternRewriter& rewriter) const;

    void setNewType(mlir::Value value, DimsOrder newOrder) const;

private:
    InputToReordersMap* const _reorders;
    Logger _log;
};

IE::ReorderOp LayerRewriter::createReorder(mlir::Operation* op, mlir::Value input, DimsOrder dstOrder,
                                           mlir::PatternRewriter& rewriter) const {
    _log.nest(2).trace("Insert Reorder: '{0}' -> '{1}'", DimsOrder::fromValue(input), dstOrder);

    const auto inputWithReordersIt = _reorders->find(input);
    if (inputWithReordersIt == _reorders->end()) {
        auto reorder = rewriter.create<IE::ReorderOp>(op->getLoc(), input, dstOrder.toAffineMap(getContext()));
        _reorders->insert({input, std::unordered_map<DimsOrder, IE::ReorderOp>{{dstOrder, reorder}}});
        return reorder;
    }

    auto currentReorders = inputWithReordersIt->getSecond();
    const auto reorderIt = currentReorders.find(dstOrder);
    if (reorderIt == currentReorders.end()) {
        auto reorder = rewriter.create<IE::ReorderOp>(op->getLoc(), input, dstOrder.toAffineMap(getContext()));
        currentReorders.insert({dstOrder, reorder});
        return reorder;
    }

    return reorderIt->second;
}

void LayerRewriter::insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                                          mlir::PatternRewriter& rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto reorderOp = createReorder(op, input.get(), dstOrder, rewriter);

    _log.nest(2).trace("Redirect input to the new Value");
    input.set(reorderOp.output());
}

void LayerRewriter::insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                           mlir::PatternRewriter& rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    auto reorderOp = createReorder(op, output, dstOrder, rewriter);

    _log.nest(2).trace("Redirect output users to the new Value");
    output.replaceAllUsesExcept(reorderOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{reorderOp});
}

void LayerRewriter::setNewType(mlir::Value val, DimsOrder newOrder) const {
    const auto origType = val.getType().cast<mlir::ShapedType>();
    const auto newType = changeDimsOrder(origType, newOrder);

    _log.nest(2).trace("Change Value type to '{0}'", newType);
    val.setType(newType);
}

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::LayoutInfoOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite layer operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto orderInfo = origOp.getLayoutInfo();
    origOp.inferLayoutInfo(orderInfo);

    rewriter.startRootUpdate(origOp);

    const auto inputs = origOp->getOpOperands();
    for (auto i : irange(inputs.size())) {
        auto& input = inputs[i];

        const auto curOrder = DimsOrder::fromValue(input.get());
        const auto supportedOrder = orderInfo.getInput(i);

        _log.nest(1).trace("Process input #{0}", i);
        _log.nest(2).trace("curOrder = {0} supportedOrder = {1}", curOrder, supportedOrder);

        if (curOrder != supportedOrder) {
            insertReorderForInput(origOp, input, supportedOrder, rewriter);
        }
    }

    const auto outputs = origOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

        _log.nest(1).trace("Process output #{0}", i);
        _log.nest(2).trace("curOrder = {0} supportedOrder = {1}", curOrder, supportedOrder);

        if (curOrder != supportedOrder) {
            setNewType(output, supportedOrder);
            insertReorderForOutput(origOp, output, curOrder, rewriter);
        }
    }

    rewriter.finalizeRootUpdate(origOp);

    return mlir::success();
}

//
// AdjustLayoutsPass
//

class AdjustLayoutsPass final : public IE::AdjustLayoutsBase<AdjustLayoutsPass> {
public:
    explicit AdjustLayoutsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustLayoutsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(op)) {
            _log.trace("Check layer operation '{0}' at '{1}'", op->getName(), op->getLoc());

            auto orderInfo = iface.getLayoutInfo();
            _log.nest().trace("Current layouts: {0}", orderInfo);

            iface.inferLayoutInfo(orderInfo);
            _log.nest().trace("Required layouts: {0}", orderInfo);

            return !orderInfo.hasChanges();
        }

        return true;
    });
    target.addLegalOp<IE::ReorderOp>();

    InputToReordersMap reorders{};
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<LayerRewriter>(&ctx, &reorders, _log.nest());

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustLayoutsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustLayoutsPass(Logger log) {
    return std::make_unique<AdjustLayoutsPass>(log);
}
