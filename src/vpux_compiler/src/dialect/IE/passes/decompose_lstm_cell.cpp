//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// LSTMCellRewriter
//

class LSTMCellRewriter final : public mlir::OpRewritePattern<IE::LSTMCellOp> {
public:
    LSTMCellRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::LSTMCellOp>(ctx), _log(log) {
        this->setDebugName("LSTMCellRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewriter::matchAndRewrite(IE::LSTMCellOp lstmCell, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", lstmCell->getName(), lstmCell->getLoc());

    auto matMulInputOp =
            rewriter.create<IE::MatMulOp>(lstmCell->getLoc(), lstmCell.inputData(), lstmCell.weights(), false, true);
    auto matMulHiddenStateOp = rewriter.create<IE::MatMulOp>(lstmCell->getLoc(), lstmCell.initialHiddenState(),
                                                             lstmCell.recurrenceWeights(), false, true);

    auto biasesAddOp = rewriter.create<IE::AddOp>(
            lstmCell->getLoc(), matMulInputOp.output(), lstmCell.biases(),
            IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY), nullptr);
    auto lstmGatesInputOp = rewriter.create<IE::AddOp>(
            lstmCell->getLoc(), biasesAddOp.output(), matMulHiddenStateOp.output(),
            IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT), nullptr);

    rewriter.replaceOpWithNewOp<IE::LSTMGatesOp>(lstmCell, lstmGatesInputOp.output(), lstmCell.initialCellState());

    return mlir::success();
}

//
// DecomposeLSTMCellPass
//

class DecomposeLSTMCellPass final : public IE::DecomposeLSTMCellBase<DecomposeLSTMCellPass> {
public:
    explicit DecomposeLSTMCellPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeLSTMCellPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LSTMCellRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeLSTMCellPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeLSTMCellPass(Logger log) {
    return std::make_unique<DecomposeLSTMCellPass>(log);
}
