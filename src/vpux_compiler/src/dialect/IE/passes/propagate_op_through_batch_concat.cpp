//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

class PropagateSoftmaxWithUnrolledMatmul final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    PropagateSoftmaxWithUnrolledMatmul(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateSoftmaxWithUnrolledMatmul");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Propagate SoftmaxOp with Unrolled MatmulOp for easier subgraph match
// when applying vertical fusion later for unet.
mlir::LogicalResult PropagateSoftmaxWithUnrolledMatmul::matchAndRewrite(IE::SoftMaxOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto reshapeOp = origOp.input().getDefiningOp<IE::ReshapeOp>();
    if (reshapeOp == nullptr) {
        _log.nest().trace("No ReshapeOp found.");
        return mlir::failure();
    }

    const auto isEnabledInput = [](mlir::Value input) {
        auto inputMatmulOp = input.getDefiningOp<IE::MatMulOp>();
        return inputMatmulOp != nullptr;
    };

    auto concatOp = reshapeOp.input().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !llvm::all_of(concatOp.getInputs(), isEnabledInput)) {
        _log.nest().trace("No valid ConcatOp found.");
        return mlir::failure();
    }

    const auto getNewSoftmaxAxis = [&](ShapeRef origShape, ShapeRef newShape,
                                       mlir::IntegerAttr origAxisIndAttr) -> mlir::IntegerAttr {
        const auto origAxisInd = getPositiveAxisInd(origAxisIndAttr, checked_cast<int64_t>(origShape.size()));
        const auto origAxisValue = origShape[Dim(origAxisInd)];
        const auto origLeftTotalSize = std::accumulate(origShape.begin(), origShape.begin() + origAxisInd,
                                                       static_cast<int64_t>(1), std::multiplies<>());
        const auto origRightTotalSize = std::accumulate(origShape.begin() + origAxisInd + 1, origShape.end(),
                                                        static_cast<int64_t>(1), std::multiplies<>());
        int64_t newLeftTotalSize = 1;
        auto newRightTotalSize =
                std::accumulate(newShape.begin(), newShape.end(), static_cast<int64_t>(1), std::multiplies<>());
        // Calculate left and right total size for each axis, if both match with original shape
        // and axis value is also the same as in the original shape, then it's the equivalent
        // softmax axis in new shape
        for (int64_t i = 0; i < static_cast<int64_t>(newShape.size()); ++i) {
            const auto newAxisValue = newShape[Dim(i)];
            // Remove current axis in right total for this iter
            newRightTotalSize /= newAxisValue;
            // If left, right total size match and axis value also match, the new axisInd is found
            if (newLeftTotalSize == origLeftTotalSize && newRightTotalSize == origRightTotalSize &&
                origAxisValue == newAxisValue) {
                return getIntAttr(getContext(), i);
            }
            // Remove current axis in left total for next iter
            newLeftTotalSize *= newAxisValue;
        }
        return nullptr;
    };

    const auto reshapeOpInShape = getShape(reshapeOp.input());
    const auto softmaxOpInShape = getShape(origOp.input());
    const auto newSoftmaxAxisAttr = getNewSoftmaxAxis(softmaxOpInShape, reshapeOpInShape, origOp.axisIndAttr());
    if (newSoftmaxAxisAttr == nullptr) {
        _log.nest().trace("Cannot find equivalent axisInd after swapping reshapeOp and softmaxOp");
        return mlir::failure();
    }

    const auto rank = reshapeOp.input().getType().dyn_cast<NDTypeInterface>().getRank();
    const auto concatAttrs = concatOp.per_axisAttr();
    if (concatAttrs == nullptr) {
        _log.nest().trace("Failed to extract concat attributes.");
        return mlir::failure();
    }

    // Concat axis must be different from softmax axis
    const auto concatAxis = getPositiveAxisInd(concatAttrs.axis(), rank);
    const auto softmaxAxis = newSoftmaxAxisAttr.getInt();
    if (concatAxis == softmaxAxis) {
        _log.nest().trace("Concat axis conflicts with softmax axis.");
        return mlir::failure();
    }

    rewriter.startRootUpdate(concatOp);
    rewriter.setInsertionPoint(concatOp);

    for (auto& concatInput : concatOp.getInputs() | indexed) {
        auto softmax = rewriter.create<IE::SoftMaxOp>(origOp.getLoc(), concatInput.value(), newSoftmaxAxisAttr);
        concatOp.setOperand(static_cast<uint32_t>(concatInput.index()), softmax.output());
    }

    rewriter.replaceOp(origOp, reshapeOp->getResults());
    rewriter.finalizeRootUpdate(concatOp);

    return mlir::success();
}

//
// PropagateOpThroughBatchConcat
//

class PropagateOpThroughBatchConcat final :
        public IE::PropagateOpThroughBatchConcatBase<PropagateOpThroughBatchConcat> {
public:
    explicit PropagateOpThroughBatchConcat(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateOpThroughBatchConcat::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateSoftmaxWithUnrolledMatmul>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateOpThroughBatchConcatPass(Logger log) {
    return std::make_unique<PropagateOpThroughBatchConcat>(log);
}
