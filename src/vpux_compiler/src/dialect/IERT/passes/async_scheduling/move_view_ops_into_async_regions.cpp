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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallPtrSet.h>

#include <algorithm>

using namespace vpux;

namespace {

//
// AsyncRegionRewriter
//

bool isPureViewOp(mlir::Operation* op) {
    return mlir::isa<mlir::ViewLikeOpInterface>(op) && !op->hasTrait<RTLayer>();
}

SmallVector<mlir::Operation*> getOuterViewLikeDeps(mlir::Block* block) {
    llvm::SmallPtrSet<mlir::Operation*, 16> outSet;

    for (auto& op : block->getOperations()) {
        for (auto arg : op.getOperands()) {
            auto* producer = arg.getDefiningOp();

            if (producer != nullptr && producer->getBlock() != block && isPureViewOp(producer)) {
                outSet.insert(producer);
            }
        }
    }

    auto outVec = to_small_vector(outSet);

    std::sort(outVec.begin(), outVec.end(), [](mlir::Operation* op1, mlir::Operation* op2) {
        return op1->isBeforeInBlock(op2);
    });

    return outVec;
}

class AsyncRegionRewriter final : public mlir::OpRewritePattern<mlir::async::ExecuteOp> {
public:
    AsyncRegionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::async::ExecuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AsyncRegionRewriter::matchAndRewrite(mlir::async::ExecuteOp execOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got 'async.execute' Operation at '{0}'", execOp->getLoc());

    auto* bodyBlock = &execOp.body().front();

    const auto outerDeps = getOuterViewLikeDeps(bodyBlock);
    if (outerDeps.empty()) {
        return matchFailed(rewriter, execOp, "The 'async.execute' inner region has no outer view-like dependencies");
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(bodyBlock);

    for (auto* origOp : outerDeps) {
        auto* newOp = rewriter.clone(*origOp);
        rewriter.replaceOpWithinBlock(origOp, newOp->getResults(), bodyBlock);
    }

    return mlir::success();
}

//
// MoveViewOpsIntoAsyncRegionsPass
//

class MoveViewOpsIntoAsyncRegionsPass final :
        public IERT::MoveViewOpsIntoAsyncRegionsBase<MoveViewOpsIntoAsyncRegionsPass> {
public:
    explicit MoveViewOpsIntoAsyncRegionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveViewOpsIntoAsyncRegionsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<AsyncRegionRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveViewOpsIntoAsyncRegionsPass
//

std::unique_ptr<mlir::Pass> vpux::IERT::createMoveViewOpsIntoAsyncRegionsPass(Logger log) {
    return std::make_unique<MoveViewOpsIntoAsyncRegionsPass>(log);
}
