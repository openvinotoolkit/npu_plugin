//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// RemoveDuplicating
//

template <typename ConcreteOp>
class RemoveDuplicating final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    RemoveDuplicating(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult RemoveDuplicating<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    ConcreteOp firstUser = origOp;
    for (auto user : origOp->getOperand(0).getUsers()) {
        if (auto currOp = mlir::dyn_cast<ConcreteOp>(user)) {
            if (currOp->isBeforeInBlock(origOp.getOperation()) && origOp.getType() == currOp.getType()) {
                firstUser = currOp;
            }
        }
    }

    for (auto user : llvm::make_early_inc_range(firstUser->getOperand(0).getUsers())) {
        if (user == firstUser) {
            continue;
        }

        if (auto currOp = mlir::dyn_cast<ConcreteOp>(user)) {
            if (firstUser.getType() == currOp.getType()) {
                // Binary Ops are duplicated only when both inputs are the same
                if (mlir::isa<IE::AddOp, IE::AndOp>(user)) {
                    const auto currOpInput1 = currOp->getOperands()[0];
                    const auto currOpInput2 = currOp->getOperands()[1];
                    const auto firstOpInput1 = firstUser->getOperands()[0];
                    const auto firstOpInput2 = firstUser->getOperands()[1];

                    const auto inputsAreEqual = (currOpInput1 == firstOpInput1) && (currOpInput2 == firstOpInput2);
                    const auto swappedInputsAreEqual =
                            (currOpInput1 == firstOpInput2) && (currOpInput1 == firstOpInput2);
                    if (!(inputsAreEqual || swappedInputsAreEqual)) {
                        continue;
                    }
                }

                _log.trace("Current node has a duplicate. Eliminate usage of current node:\n{0} {1}\n{2} {3}",
                           firstUser.getLoc(), firstUser, currOp.getLoc(), currOp);

                rewriter.replaceOp(currOp, firstUser->getResults());

                // Operation can contain the same operand in list of operands many times. For example IE.Add(%0, %0)
                // In this case, the next operation is the same as current one
                // Break loop to avoid removing current operation several times
                return mlir::success();
            }
        }
    }

    return mlir::failure();
}

//
// UniquifyOpsPass
//

class UniquifyOpsPass final : public IE::UniquifyOpsBase<UniquifyOpsPass> {
public:
    explicit UniquifyOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UniquifyOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveDuplicating<IE::ExpandOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::ReorderOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::PermuteCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::ShapeCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::QuantizeCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::AddOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::AndOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::LayoutCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::MemPermuteOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicating<IE::PermuteQuantizeOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUniquifyOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUniquifyOpsPass(Logger log) {
    return std::make_unique<UniquifyOpsPass>(log);
}
