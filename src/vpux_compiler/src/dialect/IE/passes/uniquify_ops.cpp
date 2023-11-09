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
// RemoveDuplicatingGeneric
//

template <typename ConcreteOp>
class RemoveDuplicatingGeneric : public mlir::OpRewritePattern<ConcreteOp> {
public:
    RemoveDuplicatingGeneric(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    virtual bool isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp) const;
    Logger _log;
};

template <typename ConcreteOp>
bool RemoveDuplicatingGeneric<ConcreteOp>::isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp) const {
    if (firstOp && secondOp) {
        if (firstOp.getType() == secondOp.getType()) {
            return true;
        }
    }
    return false;
}

template <typename ConcreteOp>
mlir::LogicalResult RemoveDuplicatingGeneric<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
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
            if (isDuplicatedOperation(firstUser, currOp)) {
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
// RemoveDuplicatingConcat
//

class RemoveDuplicatingConcat final : public RemoveDuplicatingGeneric<IE::ConcatOp> {
public:
    RemoveDuplicatingConcat(mlir::MLIRContext* ctx, Logger log): RemoveDuplicatingGeneric<IE::ConcatOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(IE::ConcatOp firstOp, IE::ConcatOp secondOp) const override;
};

bool RemoveDuplicatingConcat::isDuplicatedOperation(IE::ConcatOp firstOp, IE::ConcatOp secondOp) const {
    auto inputNumber = firstOp.inputs().size();
    if (inputNumber != secondOp.inputs().size()) {
        return false;
    }

    for (size_t i = 0; i < inputNumber; i++) {
        if (firstOp.inputs()[i] != secondOp.inputs()[i]) {
            return false;
        }
    }

    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    if (firstOp.per_axisAttr() != secondOp.per_axisAttr()) {
        return false;
    }

    if (firstOp.static_offsetsAttr() != secondOp.static_offsetsAttr()) {
        return false;
    }

    return true;
}

//
// RemoveDuplicatingCommutativeEltwise
//

// The class is for commutative eltwise operation like add, and, don't use it for subtract
template <typename ConcreteOp>
class RemoveDuplicatingCommutativeEltwise final : public RemoveDuplicatingGeneric<ConcreteOp> {
public:
    RemoveDuplicatingCommutativeEltwise(mlir::MLIRContext* ctx, Logger log)
            : RemoveDuplicatingGeneric<ConcreteOp>(ctx, log) {
    }

private:
    bool isDuplicatedOperation(ConcreteOp firstOp, ConcreteOp secondOp) const override;
};

template <typename ConcreteOp>
bool RemoveDuplicatingCommutativeEltwise<ConcreteOp>::isDuplicatedOperation(ConcreteOp firstOp,
                                                                            ConcreteOp secondOp) const {
    if (firstOp.getType() != secondOp.getType()) {
        return false;
    }

    const auto firstOpInput1 = firstOp->getOperands()[0];
    const auto firstOpInput2 = firstOp->getOperands()[1];
    const auto secondOpInput1 = secondOp->getOperands()[0];
    const auto secondOpInput2 = secondOp->getOperands()[1];

    const auto inputsAreEqual = (firstOpInput1 == secondOpInput1) && (firstOpInput2 == secondOpInput2);
    const auto swappedInputsAreEqual = (firstOpInput1 == secondOpInput2) && (firstOpInput2 == secondOpInput1);

    return inputsAreEqual || swappedInputsAreEqual;
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
    patterns.add<RemoveDuplicatingGeneric<IE::ExpandOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::ReorderOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::PermuteCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::ShapeCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::QuantizeCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingCommutativeEltwise<IE::AddOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingCommutativeEltwise<IE::AndOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::LayoutCastOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::MemPermuteOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingGeneric<IE::PermuteQuantizeOp>>(&ctx, _log);
    patterns.add<RemoveDuplicatingConcat>(&ctx, _log);

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
