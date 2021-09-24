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
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>
#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// GenericConverter
//

class GenericConverter final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TransposeOp>(ctx), _log(log) {
        this->setDebugName("CleanUpPermute::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp perm2, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GenericConverter::matchAndRewrite(IE::TransposeOp perm2, mlir::PatternRewriter& rewriter) const {
    if (!perm2.input().hasOneUse()) {
        return matchFailed(_log, rewriter, perm2, "perm2 is not the only user of its input Value");
    }

    auto reshapeOp = perm2.input().getDefiningOp<IE::ReshapeOp>();
    if (reshapeOp == nullptr) {
        return matchFailed(_log, rewriter, reshapeOp,
                           "reshapeOp input was not produced by another Operation or the producer does not support "
                           "post-processing");
    }

    if (!reshapeOp.input().hasOneUse()) {
        return matchFailed(_log, rewriter, reshapeOp, "reshapeOp is not the only user of its input Value");
    }

    auto perm1 = reshapeOp.input().getDefiningOp<IE::TransposeOp>();
    if (perm1 == nullptr) {
        return matchFailed(
                _log, rewriter, perm1,
                "perm1 input was not produced by another Operation or the producer does not support post-processing");
    }

    const auto outputShape = perm2.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(perm2, perm1.input(), nullptr, false, outputShapeAttr);

    return mlir::success();
}

//
// CleanUpPermutePass
//

class CleanUpPermutePass final : public IE::CleanUpPermuteBase<CleanUpPermutePass> {
public:
    CleanUpPermutePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void CleanUpPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<GenericConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createCleanUpPermutePass(Logger log) {
    return std::make_unique<CleanUpPermutePass>(log);
}
