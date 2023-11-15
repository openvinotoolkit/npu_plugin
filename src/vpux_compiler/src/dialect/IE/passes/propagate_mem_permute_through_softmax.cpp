//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// SwapSoftmaxAndMemPermute
//

class SwapSoftmaxAndMemPermute final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    SwapSoftmaxAndMemPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("SwapSoftmaxAndMemPermute");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// a sub graph in below:
//
//       SoftmaxOp
//           |
//      MemPermuteOp
//           |
//          Ops
//
// Swap the softmaxOp and MemPermuteOp
//
//      MemPermuteOp
//           |
//       SoftmaxOp
//           |
//          Ops
// If the Axis of SoftmaxOp is in the last memory dimension, the SW kernel is more efficient.

mlir::LogicalResult SwapSoftmaxAndMemPermute::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());

    auto softMaxOp = memPermuteOp.input().getDefiningOp<IE::SoftMaxOp>();
    if (softMaxOp == nullptr) {
        return mlir::failure();
    }

    const auto inType = softMaxOp.input().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    const auto axis = softMaxOp.axisInd();
    const auto inOrder = DimsOrder::fromValue(memPermuteOp.input());
    const auto outOrder = DimsOrder::fromValue(memPermuteOp.output());

    const auto pos = inOrder.dimPos(Dim(axis));
    const auto memPos = memPermuteOp.mem_perm().getDimPosition(pos);
    const auto outPos = outOrder.toDim(MemDim(memPos));

    // Check the Axis memory position, if it's on the last dimension, the SW kernel is more efficient.
    if (memPos != inRank - 1) {
        return mlir::failure();
    }

    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(memPermuteOp->getLoc(), softMaxOp.input(),
                                                             memPermuteOp.dst_order(), memPermuteOp.mem_perm());

    const auto legalizeAxisAttr = getIntAttr(rewriter.getContext(), outPos.ind());
    auto newSoftMaxOp = rewriter.replaceOpWithNewOp<IE::SoftMaxOp>(memPermuteOp, newMemPermuteOp.output(),
                                                                   legalizeAxisAttr, nullptr);
    changeDimsOrder(newSoftMaxOp, outOrder, _log.nest());
    rewriter.eraseOp(softMaxOp);

    return mlir::success();
}

//
// PropagateMemPermuteThroughSoftMaxPass
//

class PropagateMemPermuteThroughSoftMaxPass final :
        public IE::PropagateMemPermuteThroughSoftMaxBase<PropagateMemPermuteThroughSoftMaxPass> {
public:
    explicit PropagateMemPermuteThroughSoftMaxPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateMemPermuteThroughSoftMaxPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapSoftmaxAndMemPermute>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateMemPermuteThroughSoftMaxPass(Logger log) {
    return std::make_unique<PropagateMemPermuteThroughSoftMaxPass>(log);
}
