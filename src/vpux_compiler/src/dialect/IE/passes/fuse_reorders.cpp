//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// PermuteRewriter
//

class ReorderRewriter final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    ReorderRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReorderOp>(ctx), _log(log) {
        this->setDebugName("ReorderRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool lastOpPredicate(const mlir::Operation* op) {
    return mlir::dyn_cast_or_null<mlir::func::ReturnOp>(op) != nullptr;
}

bool isTrailingActShaveOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }

    const auto& opUsers = op->getUsers();
    if (!llvm::all_of(opUsers, lastOpPredicate)) {
        return false;
    }

    return IE::isActShaveKernel(op);
}

mlir::LogicalResult ReorderRewriter::matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto isEligibleForFuse = [](mlir::Operation* op) -> bool {
        // Either the NCE taks itself must be the last in the graph,
        // or it must be penultimate and followed by some activation shave task.
        return lastOpPredicate(op) || isTrailingActShaveOp(op);
    };
    if (!llvm::all_of(origOp->getUsers(), isEligibleForFuse)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation applies only to the last reorder");
    }

    if (isTrivialReorder(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderOp is actually a permute cast");
    }

    auto layerWithPermute = origOp.input().getDefiningOp<IE::LayerWithPermuteInterface>();
    if (layerWithPermute == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "ReorderRewriter applies for NCE tasks");
    }

    if (!layerWithPermute.isSupportedPermutation(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "ODU permutation does not support {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }

    if (!layerWithPermute->getResult(0).hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "ReorderRewriter applies only for NCE tasks with one consumer");
    }

    auto output = layerWithPermute->getResult(0);
    const auto origType = output.getType().cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return matchFailed(_log.nest(), rewriter, origOp, "NCE task does not implement vpux::NDTypeInterface");
    }
    const auto newType = origType.changeDimsOrder(DimsOrder::fromAffineMap(origOp.dstOrder()));
    layerWithPermute->getResult(0).setType(newType);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), layerWithPermute->getLoc());
    rewriter.replaceOp(origOp, layerWithPermute->getResult(0));

    return mlir::success();
}

//
// FuseReordersPass
//

class FuseReordersPass final : public IE::FuseReordersPassBase<FuseReordersPass> {
public:
    explicit FuseReordersPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FuseReordersPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) == 0) {
        _log.trace("FuseReordersPass is only applicable for VPUX37XX devices.");
        return;
    }

    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseReordersPass(Logger log) {
    return std::make_unique<FuseReordersPass>(log);
}
