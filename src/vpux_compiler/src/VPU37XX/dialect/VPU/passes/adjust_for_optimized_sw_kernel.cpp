//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// Common Utils
//

Dim getLowestDim(ShapeRef shape, const DimsOrder& order) {
    const auto rank = order.numDims();
    auto lowestDim = order.dimAt(rank - 1);
    for (auto idx : irange(rank)) {
        auto dim = order.dimAt(idx);
        if (shape[dim] > 1) {
            lowestDim = dim;
        }
    }
    return lowestDim;
}

int64_t getTotalSizeBeforeDim(ShapeRef shape, const DimsOrder& order, const Dim& dim) {
    int64_t totalSize = 1;
    for (auto idx : irange(order.dimPos(dim))) {
        totalSize *= shape[order.dimAt(idx)];
    }
    return totalSize;
}

//
// AdjustShapeForSoftmax
//
// This rewritter adjusts shape of softmax for optimized kernel implementations
// Supported Optimizations:
//   - Kernel optimization for softmax with axis=0 (last memdim in compiler scope)
//   - Gather dimensions on the tile dim for multishave optimizations
class AdjustShapeForSoftmax final : public mlir::OpRewritePattern<VPU::SoftMaxOp> {
public:
    AdjustShapeForSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::SoftMaxOp>(ctx), _log(log) {
        this->setDebugName("AdjustShapeForSoftmax");
    }

private:
    mlir::LogicalResult adjustForAxisZeroOpt(Shape& shape, int64_t& axisInd, const DimsOrder& order) const;
    mlir::LogicalResult adjustForMultiShaveOpt(Shape& shape, int64_t& axisInd, const DimsOrder& order,
                                               const int64_t numActShaves) const;
    mlir::LogicalResult matchAndRewrite(VPU::SoftMaxOp softmaxOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Adjusts shape of Softmax to leverage the optimized softmax kernel implementation
// for axis 0 (the last dim in compiler scope)
// Examples:
//   - Softmax(shape=[1, 16, 24, 1], axisInd=2, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 16, 1, 24], axisInd=3, layout=NCHW)
//   - Softmax(shape=[1, 1, 24, 16], axisInd=3, layout=NHWC) is adjusted to
//     Softmax(shape=[1, 16, 24, 1], axisInd=1, layout=NHWC)
// Note that these adjustments should not change the real data in memory, so this pattern
// will only be applied when axis dim is the lowest dim in memory
mlir::LogicalResult AdjustShapeForSoftmax::adjustForAxisZeroOpt(Shape& shape, int64_t& axisInd,
                                                                const DimsOrder& order) const {
    const auto axisDim = Dim(axisInd);
    const auto lowestDim = getLowestDim(shape, order);
    const auto lastDimInMem = order.dimAt(shape.size() - 1);

    if (axisDim != lowestDim || axisDim == lastDimInMem) {
        return mlir::failure();
    }

    // swap lowest dim with the last memdim
    shape[lastDimInMem] = shape[lowestDim];
    shape[lowestDim] = 1;
    // axis becomes the last memdim
    axisInd = lastDimInMem.ind();

    return mlir::success();
}

// Adjusts the shape of Softmax to leverage as much shave engines as possible by gather
// dimensions on tile dimension.
// Examples: (Assume 4 shave engines)
//   - Softmax(shape=[1, 2, 16, 24], axisInd=3, layout=NCHW) is adjusted to
//     Softmax(shape=[1, 32, 1, 24], axisInd=3, layout=NCHW)
//   - Softmax(shape=[1, 24, 2, 16], axisInd=1, layout=NHWC) is adjusted to
//     Softmax(shape=[1, 24, 32, 1], axisInd=1, layout=NHWC)
// Note that these adjustments should not change the real data in memory, and the axis dim
// should not be the tile dim
mlir::LogicalResult AdjustShapeForSoftmax::adjustForMultiShaveOpt(Shape& shape, int64_t& axisInd,
                                                                  const DimsOrder& order,
                                                                  const int64_t numActShaves) const {
    const auto axisDim = Dim(axisInd);

    // only support NCHW and NHWC layout
    if (order != DimsOrder::NCHW && order != DimsOrder::NHWC) {
        return mlir::failure();
    }

    // NCHW tile at C, NHWC tile at H
    const auto tileDim = order.dimAt(1);

    // the axis dim on or before the tile dim is not supported
    if (order.dimPos(tileDim) >= order.dimPos(axisDim)) {
        return mlir::failure();
    }

    // no need to adjust if the tile dim is large enough or
    // equal to the max possible dim shape
    const auto maxPossibleDimShape = getTotalSizeBeforeDim(shape, order, axisDim);
    if (shape[tileDim] >= numActShaves || shape[tileDim] == maxPossibleDimShape) {
        return mlir::failure();
    }

    // gather shape on the tile dim
    for (auto idx : irange(order.dimPos(axisDim))) {
        auto dim = order.dimAt(idx);
        shape[dim] = dim == tileDim ? maxPossibleDimShape : 1;
    }

    return mlir::success();
}

mlir::LogicalResult AdjustShapeForSoftmax::matchAndRewrite(VPU::SoftMaxOp softmaxOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got {0} at loc '{1}'", softmaxOp->getName(), softmaxOp->getLoc());

    const auto ctx = getContext();

    const auto inType = softmaxOp.input().getType().cast<NDTypeInterface>();
    const auto inOrder = inType.getDimsOrder();
    const auto inShape = inType.getShape();

    auto shape = inShape.toValues();
    auto axisInd = softmaxOp.axisInd();

    const auto axisZeroOpt = adjustForAxisZeroOpt(shape, axisInd, inOrder);
    if (mlir::succeeded(axisZeroOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for AxisZeroOpt", shape, axisInd);
    }

    const auto numActShaves = IE::getTotalNumOfActShaveEngines(softmaxOp->getParentOfType<mlir::ModuleOp>());
    const auto multiShaveOpt = adjustForMultiShaveOpt(shape, axisInd, inOrder, numActShaves);
    if (mlir::succeeded(multiShaveOpt)) {
        _log.nest(1).trace("Adjusted shape to {0} and axisInd to {1} for MultiShaveOpt", shape, axisInd);
    }

    if (mlir::failed(axisZeroOpt) && mlir::failed(multiShaveOpt)) {
        return mlir::failure();
    }

    auto reshapeInOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType.changeShape(shape),
                                                         softmaxOp.input(), getIntArrayAttr(ctx, shape));
    auto newSoftmaxOp = rewriter.create<VPU::SoftMaxOp>(softmaxOp.getLoc(), reshapeInOp.result(),
                                                        getIntAttr(ctx, axisInd), softmaxOp.padSizeAttr(), nullptr);
    auto reshapeOutOp = rewriter.create<VPU::ShapeCastOp>(softmaxOp.getLoc(), inType, newSoftmaxOp.output(),
                                                          getIntArrayAttr(ctx, inShape));

    softmaxOp.replaceAllUsesWith(reshapeOutOp.result());
    rewriter.eraseOp(softmaxOp);

    return mlir::success();
}

//
// AdjustForOptimizedSwKernelPass
//

class AdjustForOptimizedSwKernelPass final :
        public VPU::arch37xx::AdjustForOptimizedSwKernelBase<AdjustForOptimizedSwKernelPass> {
public:
    explicit AdjustForOptimizedSwKernelPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void AdjustForOptimizedSwKernelPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustShapeForSoftmax>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createAdjustForOptimizedSwKernelPass(Logger log) {
    return std::make_unique<AdjustForOptimizedSwKernelPass>(log);
}
