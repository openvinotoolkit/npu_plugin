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
        auto implicitOp = origOp.input().getDefiningOp<IE::SoftMaxOp>();
        if (implicitOp == nullptr) {
            return mlir::failure();
        }
        auto dimIdx = implicitOp.axisInd();
        if (dimIdx != Dims4D::Act::C.ind()) {
            return mlir::failure();
        }

        auto expandedShape = to_small_vector(getShape(origOp.output()));
        auto implicitShape = to_small_vector(getShape(implicitOp->getResult(0)));
        int64_t expandedCAxisSize = expandedShape[Dims4D::Act::C.ind()] - implicitShape[Dims4D::Act::C.ind()];
        auto optimizeSuccess = genericOptimizeSliceImplicitExpand(origOp, implicitOp.getOperation(), rewriter);
        if (optimizeSuccess.failed()) {
            return mlir::failure();
        }
        // update necessary attribute
        implicitOp.padSizeAttr(getIntAttr(rewriter.getContext(), expandedCAxisSize));
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
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::QuantizeCastOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::ConcatOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::HSwishOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSliceImplicitExpand<IE::SwishOp>>(&ctx, _log);
    patterns.add<IE::OptimizeSingleSliceConcatExpand>(&ctx, _log);
    patterns.add<OptimizeSliceSoftmaxExpand>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createOptimizeSliceExpandPass(Logger log) {
    return std::make_unique<OptimizeSliceExpandPass>(log);
}
