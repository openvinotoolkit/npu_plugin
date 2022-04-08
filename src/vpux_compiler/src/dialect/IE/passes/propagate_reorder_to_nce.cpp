//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"

#include <vpux/compiler/conversion.hpp>
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

class ActShaveRewriter final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ActShaveRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        this->setDebugName("ActShaveRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ActShaveRewriter::matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto allUsersAreReturn = llvm::all_of(origOp->getUsers(), [](auto user) {
        return mlir::isa<mlir::ReturnOp>(user);
    });
    if (!allUsersAreReturn) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder is not the last operation in the graph.");
    }

    // Check that the producer of this IE.Reorder is a software layer.
    auto producer = origOp.input().getDefiningOp();
    if (!IE::isActShaveKernel(producer)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Reorder producer is not a software layer.");
    }

    if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(producer)) {
        const auto propagatingOrder = DimsOrder::fromValue(origOp.output());

        auto orderInfo = iface.getLayoutInfo();
        orderInfo.setInput(0, propagatingOrder);
        iface.inferLayoutInfo(orderInfo);
        if (orderInfo.getInput(0) != propagatingOrder) {
            return matchFailed(_log.nest(), rewriter, producer,
                               "Act shave kernel doesn't support propagating order {0}", propagatingOrder);
        }
    }

    // Check that there is NCE task above
    auto maybeNCE = producer->getOperand(0).getDefiningOp();

    if (maybeNCE == nullptr || VPUIP::NCEInvariant::verifyKernel(maybeNCE, _log).failed()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Act shave producer is not a NCE layer.");
    }

    if (!maybeNCE->hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE operation has more than one user");
    }

    const auto dstOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto dstOrderMap = dstOrder.toAffineMap(rewriter.getContext());
    auto reorder = rewriter.create<IE::ReorderOp>(origOp->getLoc(), producer->getOperand(0), dstOrderMap);
    mlir::BlockAndValueMapping mapper;
    mapper.map(producer->getOperand(0), reorder->getResult(0));
    auto newProducer = rewriter.clone(*producer, mapper);
    vpux::inferReturnTypes(newProducer, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(producer, newProducer->getResult(0));

    return mlir::success();
}

//
// PropagateReorderToNCE
//

class PropagateReorderToNCE final : public IE::PropagateReorderToNCEBase<PropagateReorderToNCE> {
public:
    explicit PropagateReorderToNCE(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateReorderToNCE::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    auto& ctx = getContext();

    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) == 0) {
        _log.trace("PropagateReorderToNCE is only applicable for VPUX37XX device.");
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ActShaveRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateReorderToNCEPass(Logger log) {
    return std::make_unique<PropagateReorderToNCE>(log);
}
