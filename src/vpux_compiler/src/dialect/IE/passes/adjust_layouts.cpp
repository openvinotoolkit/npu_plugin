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

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    IE::ReorderOp createReorder(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                mlir::PatternRewriter& rewriter) const;

    void insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                               mlir::PatternRewriter& rewriter) const;
    void insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                mlir::PatternRewriter& rewriter) const;

    void setNewType(mlir::Value operand, DimsOrder newOrder) const;

private:
    Logger _log;
};

IE::ReorderOp LayerRewriter::createReorder(mlir::Operation* op, mlir::Value input, DimsOrder dstOrder,
                                           mlir::PatternRewriter& rewriter) const {
    _log.nest(2).trace("Create Reorder: '{0}' -> '{1}'", DimsOrder::fromValue(input), dstOrder);
    return rewriter.create<IE::ReorderOp>(op->getLoc(), input, dstOrder.toPermutationAffineMap(rewriter.getContext()));
}

void LayerRewriter::insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, DimsOrder dstOrder,
                                          mlir::PatternRewriter& rewriter) const {
    _log.nest(2).trace("Insert ReorderOp for input[{0}]", input.getOperandNumber());

    auto reorderOp = createReorder(op, input.get(), dstOrder, rewriter);
    input.set(reorderOp.output());
}

void LayerRewriter::insertReorderForOutput(mlir::Operation* op, mlir::Value output, DimsOrder dstOrder,
                                           mlir::PatternRewriter& rewriter) const {
    _log.nest(2).trace("Insert ReorderOp for output {0}", output.getType());

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    auto reorderOp = createReorder(op, output, dstOrder, rewriter);
    output.replaceAllUsesExcept(reorderOp.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{reorderOp});
}

void LayerRewriter::setNewType(mlir::Value operand, DimsOrder newOrder) const {
    const auto origType = operand.getType().cast<mlir::ShapedType>();
    const auto newType = changeDimsOrder(origType, newOrder);
    operand.setType(newType);
}

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::LayoutInfoOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto orderInfo = origOp.getLayoutInfo();
    _log.nest().trace("Current layouts: {0}", orderInfo);

    origOp.inferLayoutInfo(orderInfo);

    if (!orderInfo.hasChanges()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Current layouts are supported");
    }

    _log.nest().trace("Required layouts: {0}", orderInfo);

    rewriter.startRootUpdate(origOp);

    const auto inputs = origOp->getOpOperands();
    for (auto i : irange(inputs.size())) {
        auto& input = inputs[i];

        const auto curOrder = DimsOrder::fromValue(input.get());
        const auto supportedOrder = orderInfo.getInput(i);

        if (curOrder != supportedOrder) {
            insertReorderForInput(origOp, input, supportedOrder, rewriter);
        }
    }

    const auto outputs = origOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

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
    target.addDynamicallyLegalDialect<IE::IEDialect>([&](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(op)) {
            auto orderInfo = iface.getLayoutInfo();
            iface.inferLayoutInfo(orderInfo);
            return !orderInfo.hasChanges();
        }

        return true;
    });
    target.addLegalOp<IE::SplitOp, IE::ConcatOp, IE::ExpandOp, IE::SliceOp>();
    target.addLegalOp<IE::ReorderOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<LayerRewriter>(&ctx, _log);

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
