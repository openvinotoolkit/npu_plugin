//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/EMU/ops.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

mlir::ArrayAttr getKernelPadding(mlir::MLIRContext* ctx, VPU::PaddingAttr padAttr) {
    const auto padLeft = padAttr.getLeft().getInt();
    const auto padRight = padAttr.getRight().getInt();
    const auto padTop = padAttr.getTop().getInt();
    const auto padBottom = padAttr.getBottom().getInt();

    return getIntArrayAttr(ctx, makeArrayRef({padLeft, padRight, padTop, padBottom}));
}

//
// ConvToEMU
//

class ConvToEMU final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    ConvToEMU(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvToEMU");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvToEMU::matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder != DimsOrder::NCHW && inOrder != DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, origOp, "Operation at '{0}' has unsupported input layout '{1}'",
                           origOp->getLoc(), inOrder);
    }

    const auto isCMajor = inOrder == DimsOrder::NCHW;

    const Shape filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));
    const auto kernelPaddingAttr = getKernelPadding(getContext(), origOp.pad());
    auto taskType = isCMajor ? VPUIP::NCETaskType::CMCONV : VPUIP::NCETaskType::CONV;
    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(
            origOp->getLoc(), origOp.getType(), origOp.input(), origOp.filter(), origOp.weightsTable(), taskType,
            kernelSizeAttr, origOp.strides(), kernelPaddingAttr, origOp.rawFilterShapeAttr(), origOp.ppe());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// DepthConvToEMU
//

class DepthConvToEMU final : public mlir::OpRewritePattern<VPU::NCEDepthConvolutionOp> {
public:
    DepthConvToEMU(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEDepthConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEDepthConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DepthConvToEMU::matchAndRewrite(VPU::NCEDepthConvolutionOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder != DimsOrder::NCHW && inOrder != DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, origOp, "Operation at '{0}' has unsupported input layout '{1}'",
                           origOp->getLoc(), inOrder);
    }

    const Shape filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.rawFilterShapeAttr()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));
    const auto kernelPaddingAttr = getKernelPadding(getContext(), origOp.pad());
    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(origOp->getLoc(), origOp.getType(), origOp.input(),
                                                        origOp.filter(), origOp.weightsTable(),
                                                        VPUIP::NCETaskType::DWCONV, kernelSizeAttr, origOp.strides(),
                                                        kernelPaddingAttr, origOp.rawFilterShapeAttr(), origOp.ppe());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// MaxPoolToEMU
//

class MaxPoolToEMU final : public mlir::OpRewritePattern<VPU::NCEMaxPoolOp> {
public:
    MaxPoolToEMU(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::NCEMaxPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolToEMU::matchAndRewrite(VPU::NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto kernelPaddingAttr = getKernelPadding(getContext(), origOp.pad());
    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(
            origOp->getLoc(), origOp.getType(), origOp.input(), nullptr, origOp.weightsTable(),
            VPUIP::NCETaskType::MAXPOOL, origOp.kernel_size(), origOp.strides(), kernelPaddingAttr, origOp.ppe());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// AvgPoolToEMU
//

class AvgPoolToEMU final : public mlir::OpRewritePattern<VPU::NCEAveragePoolOp> {
public:
    AvgPoolToEMU(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::NCEAveragePoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AvgPoolToEMU::matchAndRewrite(VPU::NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto kernelPaddingAttr = getKernelPadding(getContext(), origOp.pad());
    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(origOp->getLoc(), origOp.getType(), origOp.input(), nullptr,
                                                        nullptr, VPUIP::NCETaskType::AVEPOOL, origOp.kernel_size(),
                                                        origOp.strides(), kernelPaddingAttr, origOp.ppe());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// EltwiseToEMU
//

class EltwiseToEMU final : public mlir::OpRewritePattern<VPU::NCEEltwiseOp> {
public:
    EltwiseToEMU(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::NCEEltwiseOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EltwiseToEMU::matchAndRewrite(VPU::NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(origOp.ppe().has_value(), "Eltwise operation should always have PPE info");
    auto nceOp = rewriter.create<EMU::NCEClusterTaskOp>(origOp->getLoc(), origOp.getType(), origOp.input1(),
                                                        origOp.input2(), nullptr, VPUIP::NCETaskType::ELTWISE, nullptr,
                                                        nullptr, nullptr, origOp.ppe());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// ConvertVPUNCEToEMUPass
//

class ConvertVPUNCEToEMUPass final : public ConvertVPUNCEToEMUBase<ConvertVPUNCEToEMUPass> {
public:
    explicit ConvertVPUNCEToEMUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertVPUNCEToEMUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvToEMU>(&ctx, _log);
    patterns.add<DepthConvToEMU>(&ctx, _log);
    patterns.add<MaxPoolToEMU>(&ctx, _log);
    patterns.add<AvgPoolToEMU>(&ctx, _log);
    patterns.add<EltwiseToEMU>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUNCEToEMUNCEPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUNCEToEMUPass(Logger log) {
    return std::make_unique<ConvertVPUNCEToEMUPass>(log);
}
