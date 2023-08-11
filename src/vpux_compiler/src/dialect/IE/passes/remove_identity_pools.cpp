//
// Copyright (C) 2023 Intel Corporation.
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
// RemoveIdentityPool
//

template <typename ConcreteOp>
class RemoveIdentityPool final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    RemoveIdentityPool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult RemoveIdentityPool<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());
    auto isIdentity = [](ConcreteOp origOp) {
        const auto stride = parseIntArrayAttr<int64_t>(origOp.strides());
        const auto kernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
        const auto ones = SmallVector<int64_t>(kernel.size(), 1);
        return (stride == ones && kernel == ones);
    };

    if (origOp.post_opAttr() != nullptr) {
        _log.nest().trace("Op has post_op, cannot be folded into null");
        return mlir::failure();
    }

    if (!isIdentity(origOp)) {
        _log.nest().trace("Op not identity");
        return mlir::failure();
    }

    auto inputType = origOp.input().getType();
    auto outputType = origOp.output().getType();
    if (inputType != outputType) {
        _log.nest().trace("Mismatched input/output type '{1}' with '{2}'", inputType, outputType);
        return mlir::failure();
    }

    _log.nest().trace("Replacing '{1}' with '{2}'", origOp->getName(), origOp.input());
    rewriter.replaceOp(origOp, origOp.input());
    return mlir::failure();
}

//
// RemoveIdentityPoolPass
//

class RemoveIdentityPoolPass final : public IE::RemoveIdentityPoolBase<RemoveIdentityPoolPass> {
public:
    explicit RemoveIdentityPoolPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RemoveIdentityPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveIdentityPool<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<RemoveIdentityPool<IE::AvgPoolOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// RemoveIdentityPoolPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createRemoveIdentityPoolPass(Logger log) {
    return std::make_unique<RemoveIdentityPoolPass>(log);
}
