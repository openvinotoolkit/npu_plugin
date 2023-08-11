//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <vpux/compiler/conversion.hpp>

using namespace vpux;

namespace {

//
// GenericConverter
//

template <class ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("SwapMaxPoolWithActivation::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    if (!originOp.input().hasOneUse()) {
        return matchFailed(rewriter, originOp, "Operation {0} is not the only user of its operand",
                           originOp->getName());
    }

    auto maxPool = originOp.input().template getDefiningOp<IE::MaxPoolOp>();
    if (maxPool == nullptr) {
        return matchFailed(rewriter, originOp, "Producer is not a MaxPool operation");
    }

    auto producerOp = maxPool.input().template getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return matchFailed(rewriter, originOp, "Producer of MaxPool does not support post-processing");
    }

    _log.trace("Swap MaxPool with '{0}'", originOp->getName());
    const auto activationOutType = maxPool.input().getType();
    auto newActivationOp = rewriter.create<ConcreteOp>(maxPool.getLoc(), activationOutType, maxPool.input());

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(originOp, maxPool.getType(), newActivationOp.output(),
                                               maxPool.kernel_size(), maxPool.strides(), maxPool.pads_begin(),
                                               maxPool.pads_end(), maxPool.rounding_type(), maxPool.post_opAttr());

    return mlir::success();
}

//
// SwapMaxPoolWithActivation
//

class SwapMaxPoolWithActivation final : public IE::SwapMaxPoolWithActivationBase<SwapMaxPoolWithActivation> {
public:
    explicit SwapMaxPoolWithActivation(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SwapMaxPoolWithActivation::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("SwapMaxPoolWithActivation enabled only for VPUX37XX device. Got: {0}", arch);
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter<IE::ReLUOp>>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMaxPoolWithActivation(Logger log) {
    return std::make_unique<SwapMaxPoolWithActivation>(log);
}
