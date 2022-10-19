//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPU/m2i_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// ConvertM2iResizeToTask
//

class ConvertM2iResizeToTask final : public mlir::OpRewritePattern<VPU::M2IResizeOp> {
public:
    ConvertM2iResizeToTask(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2IResizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2IResizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertM2iResizeToTask::matchAndRewrite(VPU::M2IResizeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto elType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    M2iColorFmt fmt;

    if (elType.isUnsignedInteger(8)) {
        // If last axes value is the last shape dim => planar
        const auto axes = parseIntArrayAttr<int64_t>(origOp.axes());
        const auto axesSize = axes.size();
        const auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
        if (axes[axesSize - 1] == outType.getRank() - 1) {
            fmt = M2iColorFmt::PL_RGB24;
        } else {
            fmt = M2iColorFmt::IL_RGB888;
        }
    } else if (elType.isF16()) {
        fmt = M2iColorFmt::PL_FP16_RGB;
    } else {
        VPUX_THROW("m2i unsupported format {0}", elType);
    }
    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), origOp.input(), false, false, fmt,
                                                 fmt, origOp.sizes(), origOp.axes(), nullptr);
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// ConvertM2iCscToTask
//

class ConvertM2iCscToTask final : public mlir::OpRewritePattern<VPU::M2IColorConvertOp> {
public:
    ConvertM2iCscToTask(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2IColorConvertOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2IColorConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertM2iCscToTask::matchAndRewrite(VPU::M2IColorConvertOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto iFmt = IEtoM2iColorFmt(origOp.inFmtAttr().getValue());
    auto oFmt = IEtoM2iColorFmt(origOp.outFmtAttr().getValue());

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), origOp.input(), true, false, iFmt,
                                                 oFmt, nullptr, nullptr, nullptr);
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// ConvertM2iNormToTask
//

class ConvertM2iNormToTask final : public mlir::OpRewritePattern<VPU::M2INormOp> {
public:
    ConvertM2iNormToTask(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::M2INormOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2INormOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertM2iNormToTask::matchAndRewrite(VPU::M2INormOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Build the {A,B,C,D} M2I coefs from {gamma, beta, mean, variance, eps}
    const auto gamma = parseFPArrayAttr<double>(origOp.gamma_valueAttr());
    const auto beta = parseFPArrayAttr<double>(origOp.beta_valueAttr());
    const auto mean = parseFPArrayAttr<double>(origOp.mean_valueAttr());
    const auto var = parseFPArrayAttr<double>(origOp.variance_valueAttr());
    const auto eps = origOp.epsAttr().getValueAsDouble();
    const auto numChannels = gamma.size();

    // {A,B,C,D} coefs for each channel
    std::vector<double> normCoefs;
    for (size_t i = 0; i < numChannels; i++) {
        normCoefs.push_back(gamma[i]);
        normCoefs.push_back(mean[i]);
        normCoefs.push_back(std::sqrt(var[i] + eps));
        normCoefs.push_back(beta[i]);
    }

    auto coefsAttr = getFPArrayAttr(getContext(), normCoefs);

    auto fmt = M2iColorFmt::PL_FP16_RGB;  // any planar FP16 format
    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), origOp.input(), false, true, fmt,
                                                 fmt, nullptr, nullptr, coefsAttr);
    rewriter.replaceOp(origOp, m2iOp.output());

    return mlir::success();
}

//
// ConvertM2IOpsPass
//

class ConvertM2IOpsPass final : public ConvertM2IOpsBase<ConvertM2IOpsPass> {
public:
    explicit ConvertM2IOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertM2IOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX40XX) {
        _log.trace("ConvertM2IOpsPass enabled only for VPUX4000 device. Got: {0}", arch);
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertM2iCscToTask>(&ctx, _log);
    patterns.insert<ConvertM2iResizeToTask>(&ctx, _log);
    patterns.insert<ConvertM2iNormToTask>(&ctx, _log);

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPU::M2ITaskOp>();
    target.addIllegalOp<VPU::M2IColorConvertOp>();
    target.addIllegalOp<VPU::M2IResizeOp>();
    target.addIllegalOp<VPU::M2INormOp>();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertM2IOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertM2IOpsPass(Logger log) {
    return std::make_unique<ConvertM2IOpsPass>(log);
}
