//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

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

    auto reshapeOp = origOp.getSource().getDefiningOp<IE::AffineReshapeOp>();

    auto concatOp = reshapeOp ? reshapeOp.getInput().getDefiningOp<IE::ConcatOp>()
                              : origOp.getSource().getDefiningOp<IE::ConcatOp>();
    bool hasReshape = reshapeOp != nullptr;

    if (concatOp == nullptr) {
        return mlir::failure();
    }

    if (!concatOp.getStaticOffsetsAttr()) {
        return mlir::failure();
    }

    auto sliceOffset = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    const auto sliceOffsetShape = Shape(sliceOffset);
    const auto sliceOutShape = getShape(origOp.getResult());
    auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsetsAttr());
    const auto inputs = concatOp.getInputs();
    SmallVector<vpux::ShapeRef> newInputShapes;
    SmallVector<SmallVector<int64_t>> newInputShapesVec;

    if (hasReshape) {
        const auto affineOutShape = getShape(reshapeOp.getOutput());

        const auto modifiedAxes = IE::getConcatModifiedAxis(concatOp);

        for (const auto& input : inputs) {
            const SmallVector<int64_t> newShapeVec =
                    IE::calculateInputShapeAfterSwitchConcatAndAffineReshape(input, concatOp, reshapeOp);
            newInputShapesVec.emplace_back(newShapeVec);
        }

        for (const auto& vector : newInputShapesVec) {
            newInputShapes.emplace_back(ShapeRef(vector));
        }

        auto newOffsetsAttr =
                IE::getNewConcatOffsetsParameters(concatOp.getStaticOffsetsAttr(), reshapeOp.getDimMapping(), inputs,
                                                  newInputShapes, affineOutShape, modifiedAxes);

        concatOffsets = parseIntArrayOfArrayAttr<int64_t>(newOffsetsAttr);
    } else {
        for (const auto& input : inputs) {
            newInputShapes.push_back(getShape(input));
        }
    }

    for (const auto& p : zip(inputs, concatOffsets, newInputShapes)) {
        auto curVal = std::get<0>(p);
        const auto curOffset = std::get<1>(p);
        const auto curShape = std::get<2>(p);
        const auto curOffsetShape = ShapeRef(curOffset);
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

        if (hasReshape) {
            curVal = rewriter.create<IE::AffineReshapeOp>(reshapeOp.getLoc(), curVal, reshapeOp.getDimMapping(),
                                                          getIntArrayAttr(rewriter.getContext(), curShape));
        }

        for (auto i : irange(sliceOffset.size())) {
            sliceOffset[i] -= curOffset[i];
        }

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, curVal, getIntArrayAttr(getContext(), sliceOffset),
                                                 origOp.getStaticSizes());

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
    auto func = getOperation();
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
