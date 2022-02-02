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
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// GenericConverter
//

class ConcatPostOpChange final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    ConcatPostOpChange(mlir::MLIRContext* ctx, Logger log): mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx), _log(log) {
        this->setDebugName("FusePostOps::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* postOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatPostOpChange::matchAndRewrite(mlir::Operation* postOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Eltwise operation '{1}' at '{2}'", getDebugName(), postOp->getName(), postOp->getLoc());

    if (!postOp->getOperand(0).hasOneUse()) {
        return matchFailed(_log, rewriter, postOp, "PostOp is not the only user of its input Value");
    }

    if (!mlir::isa<IE::LeakyReluOp>(postOp)) {
        return mlir::failure();
    }

    auto producerOp = postOp->getOperand(0).getDefiningOp<IE::ConcatOp>();
    if (producerOp == nullptr) {
        return matchFailed(
                _log, rewriter, postOp,
                "PostOp input was not produced by another Operation or the producer does not support post-processing");
    }

    SmallVector<int64_t> kernel = {1, 1}, strides = {1, 1}, padBegin = {0, 0}, padEnd = {0, 0};

    const auto postOpName = mlir::StringAttr::get(rewriter.getContext(), postOp->getName().getStringRef());
    const auto postOpInfo = IE::PostOp::get(postOpName, postOp->getAttrDictionary(), rewriter.getContext());

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            postOp, producerOp.output(), getIntArrayAttr(rewriter.getContext(), makeArrayRef(kernel)),
            getIntArrayAttr(rewriter.getContext(), makeArrayRef(strides)),
            getIntArrayAttr(rewriter.getContext(), makeArrayRef(padBegin)),
            getIntArrayAttr(rewriter.getContext(), makeArrayRef(padEnd)),
            IE::RoundingTypeAttr::get(rewriter.getContext(), IE::RoundingType::FLOOR), postOpInfo);

    return mlir::success();
}

//
// ConcatPostOpHandlePass
//

class ConcatPostOpHandlePass final : public IE::ConcatPostOpHandleBase<ConcatPostOpHandlePass> {
public:
    explicit ConcatPostOpHandlePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConcatPostOpHandlePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<ConcatPostOpChange>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createConcatPostOpHandle(Logger log) {
    return std::make_unique<ConcatPostOpHandlePass>(log);
}
