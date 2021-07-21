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
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// GenericConverter
//

template <class ConcreteOp>
class GenericConverter : public mlir::OpRewritePattern<ConcreteOp> {
    using PostOpAttributeGetter = std::function<mlir::ArrayRef<mlir::NamedAttribute>(ConcreteOp)>;

public:
    GenericConverter(mlir::MLIRContext* ctx, IE::PostOpKind postOp, const IE::LayerInfoDialectInterface* layerInfo,
                     Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _postOp(postOp), _layerInfo(layerInfo), _log(log) {
        this->setDebugName("FusePostOps::GenericConverter");
    }

public:
    virtual mlir::LogicalResult matchAndRewrite(ConcreteOp activationOp, mlir::PatternRewriter& rewriter) const;

protected:
    virtual mlir::SmallVector<mlir::NamedAttribute> postOpAttributes(ConcreteOp) const;

    IE::PostOpKind _postOp;
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
    auto multiLayer = mlir::dyn_cast_or_null<IE::MultiLayerInterface>(mainOp);

    if (!multiLayer || (_layerInfo && !_layerInfo->isSupportedPostProcessing(mainOp, postOp))) {
        return matchFailed(_log, rewriter, postOp,
                           "Failed to fuse PostOp, since its producer does not support post-processing");
    }

    if (multiLayer.getPostOp() != nullptr) {
        return matchFailed(_log, rewriter, postOp,
                           "Failed to fuse PostOp, since its producer already have post-processing");
    }

    const auto postOpKindAttr = IE::PostOpKindAttr::get(this->getContext(), _postOp);
    const auto postOpAttr = IE::getPostOpAttr(this->getContext(), postOpKindAttr, postOpAttributes(postOp));

    multiLayer.setPostOp(postOpAttr);
    rewriter.replaceOp(postOp, mainOp->getResult(0));

    return mlir::success();
}

template <class ConcreteOp>
mlir::SmallVector<mlir::NamedAttribute> GenericConverter<ConcreteOp>::postOpAttributes(ConcreteOp) const {
    return {};
}

class ClampConverter final : public GenericConverter<IE::ClampOp> {
public:
    ClampConverter(mlir::MLIRContext* ctx, const IE::LayerInfoDialectInterface* layerInfo, Logger log)
            : GenericConverter(ctx, IE::PostOpKind::RELUX, layerInfo, log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp activationOp, mlir::PatternRewriter& rewriter) const override final;

protected:
    mlir::SmallVector<mlir::NamedAttribute> postOpAttributes(IE::ClampOp) const override final;
};

mlir::LogicalResult ClampConverter::matchAndRewrite(IE::ClampOp postOp, mlir::PatternRewriter& rewriter) const {
    if (std::abs(postOp.minAttr().getValueAsDouble()) > std::numeric_limits<double>::epsilon()) {
        return matchFailed(_log, rewriter, postOp,
                           "Failed to fuse ClampOp like ReluX, since min value is not equal to zero");
    }
    return GenericConverter<IE::ClampOp>::matchAndRewrite(postOp, rewriter);
}

mlir::SmallVector<mlir::NamedAttribute> ClampConverter::postOpAttributes(IE::ClampOp op) const {
    return {mlir::NamedAttribute(mlir::Identifier::get("Minimum", op.getContext()), op.minAttr()),
            mlir::NamedAttribute(mlir::Identifier::get("Maximum", op.getContext()), op.maxAttr())};
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
    patterns.add<GenericConverter<IE::ReLUOp>>(&ctx, IE::PostOpKind::RELU, layerInfo, _log);
    patterns.add<ClampConverter>(&ctx, layerInfo, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFusePostOpsPass(Logger log) {
    return std::make_unique<FusePostOpsPass>(log);
}
