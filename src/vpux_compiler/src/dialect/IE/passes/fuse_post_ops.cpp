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

using namespace vpux;

namespace {

//
// GenericConverter
//

IE::PostOp getPostOpAttr(IE::ReLUOp op) {
    auto* ctx = op->getContext();

    const auto kindAttr = IE::PostOpKindAttr::get(ctx, IE::PostOpKind::RELU);
    return IE::getPostOpAttr(ctx, kindAttr);
}

IE::PostOp getPostOpAttr(IE::ClampOp op) {
    auto* ctx = op->getContext();

    const std::array<mlir::NamedAttribute, 2> params = {mlir::NamedAttribute(op.minAttrName(), op.minAttr()),
                                                        mlir::NamedAttribute(op.maxAttrName(), op.maxAttr())};

    const auto kindAttr = IE::PostOpKindAttr::get(ctx, IE::PostOpKind::CLAMP);
    return IE::getPostOpAttr(ctx, kindAttr, params);
}

template <class ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, const IE::LayerInfoDialectInterface* layerInfo, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _layerInfo(layerInfo), _log(log) {
        this->setDebugName("FusePostOps::GenericConverter");

        VPUX_THROW_UNLESS(_layerInfo != nullptr, "Got NULL pointer in {0}", this->getDebugName());
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp activationOp, mlir::PatternRewriter& rewriter) const final;

private:
    const IE::LayerInfoDialectInterface* _layerInfo = nullptr;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp postOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got PostOp '{1}' at '{2}'", this->getDebugName(), postOp->getName(), postOp->getLoc());

    if (!postOp.input().hasOneUse()) {
        return matchFailed(_log, rewriter, postOp,
                           "Failed to fuse PostOp, since it is not the only user of its input Value");
    }

    auto* mainOp = postOp.input().getDefiningOp();
    auto baseLayer = mlir::dyn_cast_or_null<IE::LayerWithPostOpInterface>(mainOp);

    if (!baseLayer || !_layerInfo->isSupportedPostProcessing(mainOp, postOp)) {
        return matchFailed(_log, rewriter, postOp,
                           "Failed to fuse PostOp, since its producer does not support post-processing");
    }

    if (baseLayer.post_op().hasValue()) {
        return matchFailed(_log, rewriter, postOp,
                           "Failed to fuse PostOp, since its producer already have post-processing");
    }

    const auto postOpAttr = getPostOpAttr(postOp);

    baseLayer.post_opAttr(postOpAttr);
    rewriter.replaceOp(postOp, mainOp->getResult(0));

    return mlir::success();
}

//
// FusePostOpsPass
//

class FusePostOpsPass final : public IE::FusePostOpsBase<FusePostOpsPass> {
public:
    FusePostOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FusePostOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto* dialect = ctx.getOrLoadDialect<IE::IEDialect>();
    VPUX_THROW_UNLESS(dialect != nullptr, "IE Dialect was not loaded");

    const auto layerInfo = dialect->getRegisteredInterface<IE::LayerInfoDialectInterface>();
    VPUX_THROW_UNLESS(layerInfo != nullptr, "LayerInfoDialect is not registered");

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<GenericConverter<IE::ReLUOp>>(&ctx, layerInfo, _log);
    patterns.add<GenericConverter<IE::ClampOp>>(&ctx, layerInfo, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFusePostOpsPass(Logger log) {
    return std::make_unique<FusePostOpsPass>(log);
}
