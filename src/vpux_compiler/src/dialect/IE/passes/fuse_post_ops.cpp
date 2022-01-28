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

class GenericConverter final : public mlir::OpTraitRewritePattern<IE::EltwiseOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpTraitRewritePattern<IE::EltwiseOp>(ctx), _log(log) {
        this->setDebugName("FusePostOps::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* postOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GenericConverter::matchAndRewrite(mlir::Operation* postOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Eltwise operation '{1}' at '{2}'", getDebugName(), postOp->getName(), postOp->getLoc());

    if (mlir::isa<IE::PReluOp>(postOp)) {
        auto pReluOp = mlir::cast<IE::PReluOp>(postOp);
        auto slopes = pReluOp.negative_slope().getDefiningOp<Const::DeclareOp>();
        const auto slopesContent = slopes.content();
        const auto slopesValue = slopesContent.getValues<double>();
        VPUX_THROW_WHEN(slopesValue.empty(), "No slope value found in PRelu");
        auto slope = slopesValue[0];

        if (llvm::all_of(slopesValue, [slope](auto const& elem) {
                return isDoubleEqual(slope, elem);
            })) {
            rewriter.replaceOpWithNewOp<IE::LeakyReluOp>(pReluOp, pReluOp.input(),
                                                         getFPAttr(pReluOp->getContext(), slope));
            return matchFailed(_log, rewriter, postOp,
                               "PRelu with same slope values is replaced by LeakyRelu for later fusing");
        }
    }

    if (!postOp->getOperand(0).hasOneUse()) {
        return matchFailed(_log, rewriter, postOp, "PostOp is not the only user of its input Value");
    }

    auto producerOp = postOp->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return matchFailed(
                _log, rewriter, postOp,
                "PostOp input was not produced by another Operation or the producer does not support post-processing");
    }
    if (!producerOp.isSupportedPostOp(postOp)) {
        return matchFailed(_log, rewriter, postOp, "PostOp producer does not support post-processing for current case");
    }
    if (producerOp.getPostOp().hasValue()) {
        return matchFailed(_log, rewriter, postOp, "PostOp producer already has post-processing '{0}'",
                           producerOp.getPostOp());
    }
    if (postOp->getNumOperands() != 1) {
        return matchFailed(_log, rewriter, postOp,
                           "Only single input operation can be attached as PostOp via attributes. Got '{0}' inputs",
                           postOp->getNumOperands());
    }

    // FIXME fuse LeakyRelu using PWL here
    const auto isQuantized = [](mlir::Operation* op, mlir::Operation* postOp) -> bool {
        auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
        auto isFakeQuantizeOpOutput = false;
        for (auto user : postOp->getUsers()) {
            if (mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(user)) {
                isFakeQuantizeOpOutput = true;
                break;
            }
        }
        return isFakeQuantizeOpOutput || isFakeQuantizeOpInput;
    };

    if (mlir::isa<IE::LeakyReluOp>(postOp) && isQuantized(producerOp, postOp)) {
        return matchFailed(_log, rewriter, postOp, "PostOp producer only support fusing Leakyrelu with FP16");
    }

    producerOp.setPostOp(postOp);
    rewriter.replaceOp(postOp, producerOp->getResult(0));

    return mlir::success();
}

//
// FusePostOpsPass
//

class FusePostOpsPass final : public IE::FusePostOpsBase<FusePostOpsPass> {
public:
    explicit FusePostOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FusePostOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<GenericConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFusePostOpsPass(Logger log) {
    return std::make_unique<FusePostOpsPass>(log);
}
