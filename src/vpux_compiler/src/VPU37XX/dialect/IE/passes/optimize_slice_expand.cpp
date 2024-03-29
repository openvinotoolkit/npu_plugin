//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes/optimize_slice_expand.hpp"
#include "vpux/compiler/VPU37XX/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// OptimizeSliceSoftmaxExpand
//

class OptimizeSliceSoftmaxExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    OptimizeSliceSoftmaxExpand(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeSliceSoftmaxExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
        const auto innerLog = _log.nest();

        auto implicitOp = origOp.getInput().getDefiningOp<IE::SoftMaxOp>();
        if (implicitOp == nullptr) {
            innerLog.trace("Expand '{0}' input is not 'SoftMaxOp'", origOp->getLoc());
            return mlir::failure();
        }

        auto dimIdx = implicitOp.getAxisInd();
        if (dimIdx != Dims4D::Act::C.ind()) {
            innerLog.trace("'SoftMaxOp' process axis should be 'Channel(1)' but got '{0}'", dimIdx);
            return mlir::failure();
        }

        auto expandedShape = to_small_vector(getShape(origOp.getOutput()));
        auto implicitShape = to_small_vector(getShape(implicitOp->getResult(0)));
        int64_t expandedCAxisSize = expandedShape[Dims4D::Act::C.ind()] - implicitShape[Dims4D::Act::C.ind()];
        const auto loc = origOp->getLoc();
        auto optimizeSuccess = genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(),
                                                                  /*hasCalculationCost=*/true, rewriter, innerLog);
        if (optimizeSuccess.failed()) {
            return mlir::failure();
        }
        // update necessary attribute
        implicitOp.setPadSizeAttr(getIntAttr(rewriter.getContext(), expandedCAxisSize));
        innerLog.trace("Optimization completed successfully at '{0}'", loc);
        return mlir::success();
    }

private:
    Logger _log;
};

//
// OptimizeSliceExpandPass
//

class OptimizeSliceExpandPass final : public IE::arch37xx::OptimizeSliceExpandBase<OptimizeSliceExpandPass> {
public:
    explicit OptimizeSliceExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeSliceExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::OptimizeSliceExpand>(&ctx, _log);
    patterns.add<IE::OptimizeExpandSlice>(&ctx, _log);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::QuantizeCastOp>>(&ctx, _log, /*hasCalculationCost=*/false);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::HSwishOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::SwishOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::GeluOp>>(&ctx, _log, /*hasCalculationCost=*/true);
    patterns.add<IE::OptimizeSliceConcatExpand>(&ctx, _log);
    patterns.add<OptimizeSliceSoftmaxExpand>(&ctx, _log);
    patterns.add<IE::OptimizeSliceTwoConcatsExpand>(&ctx, _log);

    auto func = getOperation();
    // There is case for `OptimizeExpandSlice` that the iteration time larger than 10
    // Increase the default maxIterations value from 10 to 20
    auto greedyRewriteConfig = getDefaultGreedyRewriteConfig();
    greedyRewriteConfig.maxIterations = 20;
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), greedyRewriteConfig))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createOptimizeSliceExpandPass(Logger log) {
    return std::make_unique<OptimizeSliceExpandPass>(log);
}
