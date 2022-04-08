//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// SwapMVNWithTranspose
//

class SwapMVNWithTranspose final : public IE::SwapMVNWithTransposeBase<SwapMVNWithTranspose> {
public:
    explicit SwapMVNWithTranspose(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class OpSwapConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// OpSwapConverter
//

class SwapMVNWithTranspose::OpSwapConverter final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    OpSwapConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVNOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/* This rewriter swaps MVNOp with parent TransposeOp in case:
    1.MVNOp is not cross_channel.
    2.Parent TransposeOp is swapping W and H axis.
The benifits are:
    1.Prevent inserting Reorder between TransposeOp and MVNOp.
    2.Original input and output TransposeOps have opportinuty to be elimented.
*/
mlir::LogicalResult SwapMVNWithTranspose::OpSwapConverter::matchAndRewrite(IE::MVNOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    const auto mvnIn = origOp.input();
    auto origTransposeOp = mvnIn.getDefiningOp<IE::TransposeOp>();
    auto mvnOp = rewriter.create<IE::MVNOp>(origOp->getLoc(), origTransposeOp.input(), origOp.across_channelsAttr(),
                                            origOp.normalize_varianceAttr(), origOp.epsAttr());

    rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, mvnOp.output(), nullptr, origTransposeOp.order_valueAttr());
    rewriter.eraseOp(origTransposeOp);

    return mlir::success();
}

void SwapMVNWithTranspose::safeRunOnFunc() {
    auto func = getFunction();

    auto& ctx = getContext();

    const auto isLegalOp = [](IE::MVNOp op) -> bool {
        if (!op->hasOneUse()) {
            return true;
        }

        if (op.across_channels()) {
            return true;
        }

        auto prevTranspoe = op.input().getDefiningOp<IE::TransposeOp>();
        auto nextTranspoe = mlir::dyn_cast<IE::TransposeOp>(*op.output().getUsers().begin());
        if (!prevTranspoe || !nextTranspoe || !prevTranspoe.output().hasOneUse() ||
            !nextTranspoe.output().hasOneUse()) {
            return true;
        }

        if (vpux::IE::isWHSwappingTranspose(prevTranspoe) && vpux::IE::isWHSwappingTranspose(nextTranspoe)) {
            return prevTranspoe.input().isa<mlir::BlockArgument>();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MVNOp>(isLegalOp);
    target.addLegalOp<IE::TransposeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapMVNWithTranspose::OpSwapConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMVNWithTransposePass(Logger log) {
    return std::make_unique<SwapMVNWithTranspose>(log);
}
