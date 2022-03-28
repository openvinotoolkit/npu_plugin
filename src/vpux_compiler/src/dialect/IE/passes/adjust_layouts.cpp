//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/adjust_layout_utils.hpp"
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
    LayerRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

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
        if (curOrder != supportedOrder) {
            insertReorderForInput(origOp, input, supportedOrder, rewriter, _log.nest());
        }
    }

    const auto outputs = origOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

        _log.nest(1).trace("Process output #{0}", i);
        if (curOrder != supportedOrder) {
            changeDimsOrder(output, supportedOrder, _log.nest());
            insertReorderForOutput(origOp, output, curOrder, rewriter, _log.nest());
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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<LayerRewriter>(&ctx, _log.nest());

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
