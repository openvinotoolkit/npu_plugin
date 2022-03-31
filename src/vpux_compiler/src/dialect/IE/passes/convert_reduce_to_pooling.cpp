//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

struct PoolingAttr {
    mlir::ArrayAttr kernelAttr;
    mlir::ArrayAttr stridesAttr;
    mlir::ArrayAttr padBeginAttr;
    mlir::ArrayAttr padEndAttr;
    IE::RoundingTypeAttr roundingAttr;
    mlir::UnitAttr excludePadsAttr;
};

bool isReduceOneDim(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> axes) {
    for (const auto& axis : axes) {
        if (inputShape[axis] != 1) {
            return false;
        }
    }
    return true;
}

bool isSpatialDimsReduction(ArrayRef<int64_t> axes) {
    for (const auto& axis : axes) {
        if (axis <= 1) {
            return false;
        }
    }
    return true;
}

void constructAvgOpParams(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape, ArrayRef<int64_t> axes,
                          SmallVector<int64_t>& kernel, SmallVector<int64_t>& strides, SmallVector<int64_t>& padBegin,
                          SmallVector<int64_t>& padEnd, SmallVector<int64_t>& shapeBegin,
                          SmallVector<int64_t>& shapeEnd) {
    /*
     * Prepare default attributes for Pooling operation
     *      padBegin/padEnd - should be zeros as we don't need any padding
     *      strides - should be filled with ones
     *      kernel  - depends on Reduction operation axes
     *
     * Also here we decide should we use Reshapes before and after Pooling
     *      shapeBegin - if not empty indicates that we need a Reshape before Pooling
     *      shapeEnd   - if not empty indicates that we need a Reshape after Pooling
     */

    shapeEnd = to_small_vector(outputShape);
    strides.assign({1, 1});
    padBegin.assign({0, 0});
    padEnd.assign({0, 0});

    if (!isSpatialDimsReduction(axes) || inputShape.size() != 4) {
        // In case if reduction applies not to spatial dimensions
        // we have to fit it into 4D Pooling
        int64_t dimsProd = 1;
        int64_t dimsBegin = 1;
        int64_t dimsEnd = 1;
        for (int64_t i = 0; static_cast<size_t>(i) < inputShape.size(); ++i) {
            if (i < axes.front()) {
                dimsBegin *= checked_cast<size_t>(inputShape[i]);
            } else if (i >= axes.front() && i <= axes.back()) {
                dimsProd *= inputShape[i];
            } else {
                dimsEnd *= inputShape[i];
            }
        }
        // The batch dimension is repositioned in the shape
        // only in case of batch dimension reduction
        shapeBegin.assign({dimsBegin, 1, dimsProd, dimsEnd});
        kernel.assign({dimsProd, 1});
    } else {
        // When input shape size is 4 and axis is not 0 or 1
        shapeBegin = to_small_vector(inputShape);
        kernel.assign({1, 1});
        for (const auto& axis : axes) {
            kernel[axis - 2] = inputShape[axis];
        }
    }
}

mlir::LogicalResult generalReduceRewrite(
        mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
        FuncRef<mlir::Operation*(mlir::Location, mlir::Value, PoolingAttr)> makeOperations) {
    const auto inputShape = getShape(origOp->getOperand(0)).raw();
    const auto outputShape = getShape(origOp->getResult(0)).raw();
    auto valueConst = origOp->getOperand(1).getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(valueConst != nullptr, "Failed to get axes in Reduce operation");
    const auto valueContent = valueConst.content();
    const auto axes = to_small_vector(valueContent.getValues<int64_t>());
    auto* ctx = origOp->getContext();

    // If Reduce op reduces only 1 dims we replace it with Reshape
    if (isReduceOneDim(inputShape, axes)) {
        auto newResult = origOp->getOperand(0);
        if (inputShape != outputShape) {
            const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
            newResult = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp->getOperand(0), nullptr, false,
                                                       outputShapeAttr);
        }
        rewriter.replaceOp(origOp, newResult);
        return mlir::success();
    }

    SmallVector<int64_t> kernel, strides, padBegin, padEnd, shapeBegin, shapeEnd;
    constructAvgOpParams(inputShape, outputShape, axes, kernel, strides, padBegin, padEnd, shapeBegin, shapeEnd);

    /*
     *  ReduceMean => AvgPool
     *                AvgPool->Reshape (in case if keep_dims=False)
     *                Reshape->AvgPool->Reshape (in case when axes don't match spatial dims)
     *  ReduceMax  => MaxPool
     *                MaxPool->Reshape (in case if keep_dims=False)
     *                Reshape->MaxPool->Reshape (in case when axes don't match spatial dims)
     *  ReduceSum  => AvgPool->Multiply
     *                AvgPool->Multiply->Reshape (in case if keep_dims=False)
     *                Reshape->AvgPool->Multiply->Reshape (in case when axes don't match spatial dims)
     *  ReduceMin  => Negative->MaxPool->Negative
     *                Negative->MaxPool->->Negative->Reshape (in case if keep_dims=False)
     *                Reshape->Negative->MaxPool->Negative->Reshape (in case when axes don't match spatial dims)
     */

    auto input = origOp->getOperand(0);

    if (shapeBegin != inputShape) {
        const auto shapeBeginAttr = getIntArrayAttr(ctx, shapeBegin);
        input = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), input, nullptr, false, shapeBeginAttr);
    }

    const auto kernelAttr = getIntArrayAttr(ctx, makeArrayRef(kernel));
    const auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef(strides));
    const auto padBeginAttr = getIntArrayAttr(ctx, makeArrayRef(padBegin));
    const auto padEndAttr = getIntArrayAttr(ctx, makeArrayRef(padEnd));
    const auto roundingAttr = IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR);
    const auto excludePadsAttr = mlir::UnitAttr::get(ctx);

    auto* newOp = makeOperations(origOp->getLoc(), input,
                                 {kernelAttr, stridesAttr, padBeginAttr, padEndAttr, roundingAttr, excludePadsAttr});
    input = newOp->getResult(0);

    if (shapeEnd != to_small_vector(getShape(input))) {
        const auto shapeEndAttr = getIntArrayAttr(ctx, shapeEnd);
        input = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), input, nullptr, false, shapeEndAttr);
    }

    rewriter.replaceOp(origOp, input);
    return mlir::success();
}

//
// ReduceMeanRewriter
//

class ReduceMeanRewriter final : public mlir::OpRewritePattern<IE::ReduceMeanOp> {
public:
    ReduceMeanRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMeanOp>(ctx), _log(log) {
        setDebugName("ReduceMeanRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMeanOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMeanRewriter::matchAndRewrite(IE::ReduceMeanOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMean layer at '{1}'", getDebugName(), origOp->getLoc());
    return generalReduceRewrite(
            origOp, rewriter, [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                return rewriter.create<IE::AvgPoolOp>(loc, input, attr.kernelAttr, attr.stridesAttr, attr.padBeginAttr,
                                                      attr.padEndAttr, attr.roundingAttr, attr.excludePadsAttr);
            });
}

//
// ReduceMaxRewriter
//

class ReduceMaxRewriter final : public mlir::OpRewritePattern<IE::ReduceMaxOp> {
public:
    ReduceMaxRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMaxOp>(ctx), _log(log) {
        setDebugName("ReduceMaxRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMaxRewriter::matchAndRewrite(IE::ReduceMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMax layer at '{1}'", getDebugName(), origOp->getLoc());
    return generalReduceRewrite(
            origOp, rewriter, [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                return rewriter.create<IE::MaxPoolOp>(loc, input, attr.kernelAttr, attr.stridesAttr, attr.padBeginAttr,
                                                      attr.padEndAttr, attr.roundingAttr, nullptr);
            });
}

//
// ReduceSumRewriter
//

class ReduceSumRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    ReduceSumRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("ReduceSumRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceSumRewriter::matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origOp->getLoc());
    return generalReduceRewrite(
            origOp, rewriter, [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                input = rewriter.create<IE::AvgPoolOp>(loc, input, attr.kernelAttr, attr.stridesAttr, attr.padBeginAttr,
                                                       attr.padEndAttr, attr.roundingAttr, attr.excludePadsAttr);

                mlir::MLIRContext* ctx = origOp->getContext();
                const auto dataStorageTensor = mlir::RankedTensorType::get({1}, mlir::Float16Type::get(ctx));
                const auto inputShape = getShape(origOp.input()).raw();
                auto valueConst = origOp.axes().getDefiningOp<Const::DeclareOp>();
                VPUX_THROW_UNLESS(valueConst != nullptr, "Failed to get axes in Reduce operation");
                const auto valueContent = valueConst.content();
                const auto axes = valueContent.getValues<int64_t>();
                float reductionDimsCount = 1;
                for (const auto& axis : axes) {
                    reductionDimsCount *= inputShape[axis];
                }
                const auto reductionDimsCountFP16 = static_cast<ov::float16>(reductionDimsCount);
                const auto baseAttr = mlir::DenseElementsAttr::get(dataStorageTensor, reductionDimsCountFP16);
                auto cst = rewriter.create<Const::DeclareOp>(loc, dataStorageTensor, Const::ContentAttr::get(baseAttr));

                const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
                return rewriter.create<IE::MultiplyOp>(loc, input, cst.output(), broadCastAttr, nullptr);
            });
}

//
// ReduceMinRewriter
//

class ReduceMinRewriter final : public mlir::OpRewritePattern<IE::ReduceMinOp> {
public:
    ReduceMinRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMinOp>(ctx), _log(log) {
        setDebugName("ReduceMinRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMinRewriter::matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMin layer at '{1}'", getDebugName(), origOp->getLoc());
    return generalReduceRewrite(
            origOp, rewriter, [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                auto scale1 = rewriter.create<IE::NegativeOp>(loc, input.getType(), input);
                auto maxPool =
                        rewriter.create<IE::MaxPoolOp>(loc, scale1.output(), attr.kernelAttr, attr.stridesAttr,
                                                       attr.padBeginAttr, attr.padEndAttr, attr.roundingAttr, nullptr);
                return rewriter.create<IE::NegativeOp>(loc, maxPool.output().getType(), maxPool.output());
            });
}

//
// ConvertReduceToPoolingPass
//

class ConvertReduceToPoolingPass final : public IE::ConvertReduceToPoolingBase<ConvertReduceToPoolingPass> {
public:
    explicit ConvertReduceToPoolingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertReduceToPoolingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalOp = [&](mlir::Operation* op) {
        const auto inputShape = getShape(op->getOperand(0)).raw();

        // Check that axes are consecutive otherwise this conversion is not applicable
        auto valueConst = op->getOperand(1).getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_UNLESS(valueConst != nullptr, "Failed to get axes in Reduce operation");
        const auto valueContent = valueConst.content();
        auto axes = to_small_vector(valueContent.getValues<int64_t>());
        for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] < 0) {
                axes[i] += inputShape.size();
            }
        }
        std::sort(axes.begin(), axes.end());
        for (size_t i = 1; i < axes.size(); i++) {
            if (axes[i] - axes[i - 1] != 1) {
                return true;
            }
        }

        auto module = getOperation();
        const auto arch = VPU::getArch(module);
        if (arch == VPU::ArchKind::VPUX30XX || arch == VPU::ArchKind::VPUX311X) {
            // Check that axis dimensions <= 255 otherwise this conversion is not applicable
            bool upaCompatible = true;
            int64_t mergedDim = 1;
            for (const auto& axis : axes) {
                mergedDim *= inputShape[axis];
                if (inputShape[axis] > 255 || mergedDim > 255) {
                    upaCompatible = false;
                }
            }

            // Check that handleLargeKernels supports this op
            const bool dpuCompatible = (axes.size() == 2) ? (vpux::IE::isPoolingKernelSizeValid(inputShape[axes[0]]) &&
                                                             vpux::IE::isPoolingKernelSizeValid(inputShape[axes[1]]))
                                                          : vpux::IE::isPoolingKernelSizeValid(mergedDim);

            const bool isHWCompilationMode = VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceHW ||
                                             VPU::getCompilationMode(op) == VPU::CompilationMode::DefaultHW;

            return !((upaCompatible && !isHWCompilationMode) || (dpuCompatible && isHWCompilationMode));
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::AvgPoolOp>();
    target.addLegalOp<IE::MaxPoolOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addDynamicallyLegalOp<IE::ReduceMeanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMaxOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceSumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMinOp>(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ReduceMeanRewriter>(&ctx, _log);
    patterns.insert<ReduceMaxRewriter>(&ctx, _log);
    patterns.insert<ReduceSumRewriter>(&ctx, _log);
    patterns.insert<ReduceMinRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReduceToPoolingPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReduceToPoolingPass(Logger log) {
    return std::make_unique<ConvertReduceToPoolingPass>(log);
}
