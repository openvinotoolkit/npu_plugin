//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ColorConvertToM2I
//

class ColorConvertToM2I final : public mlir::OpRewritePattern<IE::YuvToRgbOp> {
public:
    ColorConvertToM2I(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::YuvToRgbOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::YuvToRgbOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ColorConvertToM2I::matchAndRewrite(IE::YuvToRgbOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::M2IColorConvertOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2IColorConvertOp>(origOp->getLoc(), origOp.getType(), origOp.input1(),
                                                         origOp.inFmtAttr(), origOp.outFmtAttr());
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// InterpolateToM2I
//

class InterpolateToM2I final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateToM2I(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InterpolateToM2I::matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::M2IResizeOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2IResizeOp>(origOp->getLoc(), origOp.getType(), origOp.input(),
                                                   origOp.sizes_attrAttr(), origOp.axes_attrAttr());
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// BatchNormToM2I
//

class BatchNormToM2I final : public mlir::OpRewritePattern<IE::BatchNormInferenceOp> {
public:
    BatchNormToM2I(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::BatchNormInferenceOp>(ctx, vpux::benefitMid), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::BatchNormInferenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BatchNormToM2I::matchAndRewrite(IE::BatchNormInferenceOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::M2INormOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2INormOp>(
            origOp->getLoc(), origOp.getType(), origOp.input(), origOp.gamma_valueAttr(), origOp.beta_valueAttr(),
            origOp.mean_valueAttr(), origOp.variance_valueAttr(), origOp.epsAttr());

    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// ConvertIEToVPUM2IPass
//

class ConvertIEToVPUM2IPass final : public ConvertIEToVPUM2IBase<ConvertIEToVPUM2IPass> {
public:
    explicit ConvertIEToVPUM2IPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertIEToVPUM2IPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX40XX) {
        _log.trace("Convert to VPU-M2I Pass enabled only for VPUX4000 device. Got: {0}", arch);
        return;
    }

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<ColorConvertToM2I>(&ctx, _log);
    patterns.add<InterpolateToM2I>(&ctx, _log);
    patterns.add<BatchNormToM2I>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUM2IPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIEToVPUM2IPass(Logger log) {
    return std::make_unique<ConvertIEToVPUM2IPass>(log);
}
