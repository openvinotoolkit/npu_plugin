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

Optional<Dim> getBatchDim(ShapeRef shape) {
    Optional<Dim> batchDim = None;
    switch (shape.size()) {
    case 4:
        // batch dim is at position 1 for 4d shape when dim 0 is 1
        if (shape[Dim(0)] == 1) {
            batchDim = Dim(1);
        }
        break;
    case 3:
    case 2:
        // batch dim is at position 0 for 3d/2d shape
        batchDim = Dim(0);
        break;
    default:
        batchDim = None;
        break;
    }
    return batchDim;
}

bool isBatchConcat(IE::ConcatOp concatOp) {
    const auto concatAttrs = concatOp.per_axisAttr();
    if (concatAttrs == nullptr) {
        return false;
    }

    const auto outputType = concatOp.output().getType().dyn_cast<NDTypeInterface>();
    const auto rank = outputType.getRank();
    const auto concatAxis = getPositiveAxisInd(concatAttrs.getAxis(), rank);
    const auto batchDim = getBatchDim(outputType.getShape());
    if (!batchDim.has_value()) {
        return false;
    }
    if (concatAxis != batchDim.value().ind()) {
        return false;
    }

    const auto concatInputs = concatOp.getInputs();
    if (concatInputs.size() == 0) {
        return false;
    }
    const auto firstShape = getShape(concatInputs.front());
    return llvm::all_of(concatInputs, [&](const mlir::Value v) {
        return getShape(v) == firstShape;
    });
}

// Propagate SoftmaxOp with Unrolled MatmulOp for easier subgraph match when
// applying vertical fusion later for vertical graph "matmul->softmax->matmul"
// Need to generalize this method: E#80881
class PropagateSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    PropagateSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (origOp.input().isa<mlir::BlockArgument>()) {
        return matchFailed(_log, rewriter, origOp, "Input of SoftmaxOp is block argument");
    }

    const auto isEnabledInput = [](mlir::Value input) {
        auto inputOp = input.getDefiningOp();
        while (mlir::isa_and_nonnull<IE::ReshapeOp>(inputOp)) {
            if (!inputOp->hasOneUse()) {
                return false;
            }
            inputOp = inputOp->getOperand(0).getDefiningOp();
        }
        return mlir::isa_and_nonnull<IE::MatMulOp>(inputOp) && inputOp->hasOneUse();
    };

    auto maybeAddOp = origOp.input().getDefiningOp<IE::AddOp>();
    if (maybeAddOp != nullptr && (maybeAddOp.input2().getDefiningOp<Const::DeclareOp>() == nullptr ||
                                  getShape(maybeAddOp.input2()).totalSize() != 1)) {
        return matchFailed(_log, rewriter, origOp, "Found invalid AddOp before SoftmaxOp");
    }

    auto concatOp = maybeAddOp == nullptr ? origOp.input().getDefiningOp<IE::ConcatOp>()
                                          : maybeAddOp.input1().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !isBatchConcat(concatOp) || !llvm::all_of(concatOp.getInputs(), isEnabledInput)) {
        return matchFailed(_log, rewriter, origOp, "No valid ConcatOp found");
    }

    // Concat axis must be different from softmax axis
    const auto rank = origOp.input().getType().dyn_cast<NDTypeInterface>().getRank();
    const auto concatAttrs = concatOp.per_axisAttr();
    const auto concatAxis = getPositiveAxisInd(concatAttrs.getAxis(), rank);
    const auto softmaxAxis = getPositiveAxisInd(origOp.axisIndAttr(), rank);
    if (concatAxis == softmaxAxis) {
        return matchFailed(_log, rewriter, origOp, "Concat axis conflicts with softmax axis");
    }

    rewriter.startRootUpdate(concatOp);
    rewriter.setInsertionPoint(concatOp);

    for (auto& concatInput : concatOp.getInputs() | indexed) {
        auto sliceSoftmaxInput =
                maybeAddOp == nullptr
                        ? concatInput.value()
                        : rewriter.create<IE::AddOp>(maybeAddOp.getLoc(), concatInput.value(), maybeAddOp.input2(),
                                                     maybeAddOp.auto_broadcastAttr(), maybeAddOp.post_opAttr());
        auto sliceSoftmaxOp = rewriter.create<IE::SoftMaxOp>(origOp.getLoc(), sliceSoftmaxInput, origOp.axisIndAttr(),
                                                             origOp.padSizeAttr());
        concatOp.setOperand(checked_cast<uint32_t>(concatInput.index()), sliceSoftmaxOp.output());
    }

    rewriter.replaceOp(origOp, concatOp->getResults());
    rewriter.finalizeRootUpdate(concatOp);

    return mlir::success();
}

//
// PropagateReshape
//
class PropagateReshape final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    PropagateReshape(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReshapeOp>(ctx), _log(log) {
        this->setDebugName("PropagateOpThroughBatchConcat::PropagateReshape");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateReshape::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (origOp.input().isa<mlir::BlockArgument>()) {
        return matchFailed(_log, rewriter, origOp, "Input of ReshapeOp is block argument");
    }

    auto concatOp = origOp.input().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || !concatOp->hasOneUse() || !isBatchConcat(concatOp)) {
        return matchFailed(_log, rewriter, origOp, "ConcatOp not found or invalid");
    }

    const auto inputShape = getShape(origOp.input());
    if (inputShape.size() != 2) {
        return matchFailed(_log, rewriter, origOp, "Unsupported input shape: {0}", inputShape);
    }

    const auto outputShape = getShape(origOp.output());
    const auto batchDim = getBatchDim(outputShape);
    if (!batchDim.has_value()) {
        return matchFailed(_log, rewriter, origOp, "Unsupported output shape: {0}", outputShape);
    }

    auto sliceOutShape4D = outputShape.toValues();
    sliceOutShape4D[batchDim.value()] = 1;

    const auto concatInputs = concatOp.getInputs();
    const auto concatInputShape = getShape(concatInputs.front());

    VPUX_THROW_WHEN(concatInputShape.totalSize() != sliceOutShape4D.totalSize(),
                    "Size of inferred 4D shape of concat input ({0}) not match with original shape ({1})",
                    sliceOutShape4D, concatInputShape);

    _log.nest().trace("Propagating ReshapeOp before batch ConcatOp");

    const auto sliceOutShape4DAttr = getIntArrayAttr(rewriter.getContext(), sliceOutShape4D.raw());

    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& concatInput : concatInputs) {
        auto sliceReshape4D =
                rewriter.create<IE::ReshapeOp>(origOp.getLoc(), concatInput, nullptr, false, sliceOutShape4DAttr);
        _log.nest(2).trace("Inserted ReshapeOp: {0}", sliceReshape4D);
        newConcatInputs.push_back(sliceReshape4D.output());
    }

    auto newConcatOp = rewriter.create<IE::ConcatOp>(concatOp->getLoc(), newConcatInputs, batchDim.value());
    rewriter.replaceOp(origOp, newConcatOp.output());

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
    patterns.add<PropagateReshape>(&ctx, _log);
    patterns.add<PropagateSoftmax>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateOpThroughBatchConcatPass(Logger log) {
    return std::make_unique<PropagateOpThroughBatchConcat>(log);
}
