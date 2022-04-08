//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConcatSliceRewriter
//

class ConcatSliceRewriter final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ConcatSliceRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ConcatSliceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatSliceRewriter::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite Slice operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto concatOp = origOp.source().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    if (!concatOp.static_offsetsAttr()) {
        return mlir::failure();
    }

    auto sliceOffset = parseIntArrayAttr<int64_t>(origOp.static_offsets());
    const auto sliceOffsetShape = Shape(sliceOffset);
    const auto sliceOutShape = getShape(origOp.result());
    const auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.static_offsetsAttr());

    for (const auto& p : zip(concatOp.inputs(), concatOffsets)) {
        const auto curVal = std::get<0>(p);
        const auto curShape = getShape(curVal);
        const auto curOffset = std::get<1>(p);
        const auto curOffsetShape = Shape(curOffset);

        auto isSubTensor = [&]() -> bool {
            for (const auto ind : irange(sliceOutShape.size())) {
                const auto d = Dim(ind);

                if ((sliceOffsetShape[d] < curOffsetShape[d]) ||
                    (curOffsetShape[d] + curShape[d] < sliceOffsetShape[d] + sliceOutShape[d])) {
                    return false;
                }
            }

            return true;
        };

        if (!isSubTensor()) {
            continue;
        }

        _log.trace("ConcatSliceRewriter '{0}' at '{1}', {2}->{3}, {4}->{5}", origOp->getName(), origOp->getLoc(),
                   sliceOffsetShape, curOffsetShape, sliceOutShape, curShape);

        for (auto i : irange(sliceOffset.size())) {
            sliceOffset[i] -= curOffset[i];
        }

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, curVal, getIntArrayAttr(getContext(), sliceOffset),
                                                 origOp.static_sizes());

        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeConcatSlicePass
//

class OptimizeConcatSlicePass final : public IE::OptimizeConcatSliceBase<OptimizeConcatSlicePass> {
public:
    explicit OptimizeConcatSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeConcatSlicePass::safeRunOnFunc() {
    auto func = getFunction();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConcatSliceRewriter>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeConcatSlicePass(Logger log) {
    return std::make_unique<OptimizeConcatSlicePass>(log);
}
