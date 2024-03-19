//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// SplitInterpolate
//

class SplitInterpolate final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    SplitInterpolate(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::InterpolateOp>(ctx), _log(log) {
        setDebugName("SplitInterpolate");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalAndBenifitSplitInterpolate(VPU::InterpolateOp origOp, NDTypeInterface inputType,
                                           NDTypeInterface outputType, VPU::NCEInterpolateModeAttr modeAttr,
                                           IE::InterpolateCoordModeAttr coordModeAttr) const;

    Logger _log;
};

bool SplitInterpolate::isLegalAndBenifitSplitInterpolate(VPU::InterpolateOp origOp, NDTypeInterface inputType,
                                                         NDTypeInterface outputType,
                                                         VPU::NCEInterpolateModeAttr modeAttr,
                                                         IE::InterpolateCoordModeAttr coordModeAttr) const {
    const auto inputElemType = inputType.getElementType();
    // If NCEInterpolate has a quantized type, splitting might cause accuracy issues
    if (inputElemType.isa<mlir::quant::QuantizedType>()) {
        return false;
    }

    auto potentialScales = VPU::getNCEInterpolateScales(inputType, outputType, coordModeAttr);
    VPUX_THROW_UNLESS(potentialScales.has_value(), "Cannot get scales of NCE Interpolate");
    const auto scales = potentialScales.value();

    const auto factors = VPU::getNCEInterpolateFactors(scales, modeAttr, coordModeAttr);
    const auto padsBegin = VPU::getNCEInterpolatePadsBegin(scales, modeAttr, coordModeAttr);
    const auto padsEnd = VPU::getNCEInterpolatePadsEnd(scales, modeAttr, coordModeAttr);
    const auto inShape = inputType.getShape();

    // There is no precision cost function to guide when split is beneficial
    // Interpolate shape size and factor size directly affects performance
    // Here is a rough limitation when the NCEInterpolate size is larger than the CMX size
    // and both factors on H and W are larger than 4, it is beneficial to split Interpolate.
    const auto sepType = mlir::IntegerType::get(origOp.getContext(), 32);
    const auto seTableH = inShape[Dims4D::Act::H] * factors[VPU::SE_INTERPOLATE_FACTOR_H] +
                          padsBegin[Dims4D::PadsBegin::Top.ind()] + padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    const auto seTableW = inShape[Dims4D::Act::W] * factors[VPU::SE_INTERPOLATE_FACTOR_W] +
                          padsBegin[Dims4D::PadsBegin::Left.ind()] + padsEnd[Dims4D::PadsEnd::Right.ind()];
    auto arch = VPU::getArch(origOp);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    const int64_t seSize = VPU::getSESize(inShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inShape[Dims4D::Act::C] / seSize;

    const auto inputDataSize = inputType.getTotalAllocSize().count();
    const auto inputSMSize = (seTableH * seTableW * inShape[Dims4D::Act::C]) / CHAR_BIT;
    const auto inputSESize = (seTableH * seTableW * seDepth * sepType.getWidth()) / CHAR_BIT;
    const auto outputSize = outputType.getTotalAllocSize().count();
    const auto requiredCMX = inputDataSize + inputSMSize + inputSESize + outputSize;
    const auto doesFitIntoCMX = (requiredCMX < VPU::getTotalCMXSize(origOp).count());

    const auto areFactorsLarge = llvm::all_of(factors, [](const auto factor) {
        return factor >= 4;
    });
    return !doesFitIntoCMX && areFactorsLarge;
}

mlir::LogicalResult SplitInterpolate::matchAndRewrite(VPU::InterpolateOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (!VPU::NCEInterpolateOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return matchFailed(rewriter, origOp, "It is not NCEInterpolateOp");
    }

    const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto modeAttr = VPU::getNCEInterpolateModeAttr(origOp.getAttr().getMode());
    const auto coordModeAttr = origOp.getAttr().getCoordMode();

    if (!isLegalAndBenifitSplitInterpolate(origOp, inputType, outputType, modeAttr, coordModeAttr)) {
        return matchFailed(rewriter, origOp, "It is not beneficial to split");
    }

    const auto sizesAttr = origOp.getSizesAttrAttr();
    const auto scalesAttr = origOp.getScalesAttrAttr();
    const auto axes = IE::getInterpAxesVal(origOp.getLoc(), origOp.getAxes(), origOp.getAxesAttrAttr(), inputType);
    const auto shapeCalcMode = origOp.getAttr().getShapeCalcMode().getValue();

    auto createSingleDimInterpOp = [&](Dim dim, mlir::Value inputVal) {
        auto dimPtr = std::find(axes.begin(), axes.end(), dim.ind());
        VPUX_THROW_WHEN(dimPtr == axes.end(), "Cannot find Dim [{0}] in Interpolate axes attribution [{1}]", dim, axes);
        auto dimIdx = std::distance(axes.begin(), dimPtr);

        auto newSizesAttr = sizesAttr;
        auto newScalesAttr = scalesAttr;
        if (shapeCalcMode == IE::InterpolateCalcMode::SIZES) {
            const auto inputShape = getShape(inputVal);
            const auto sizes = parseIntArrayAttr<int64_t>(sizesAttr);
            auto newSizes = SmallVector<double>(sizes.size(), 1.0);
            for (auto axis : axes | indexed) {
                newSizes[axis.index()] = dim == Dim(axis.value()) ? sizes[axis.index()] : inputShape[Dim(axis.value())];
            }
            newSizesAttr = getIntArrayAttr(getContext(), newSizes);
        }

        if (shapeCalcMode == IE::InterpolateCalcMode::SCALES) {
            const auto scales = parseFPArrayAttr<double>(scalesAttr);
            auto newScales = SmallVector<double>(scales.size(), 1.0);
            newScales[dimIdx] = scales[dimIdx];
            newScalesAttr = getFPArrayAttr(getContext(), newScales);
        }

        auto newLoc = appendLoc(origOp.getLoc(), "_interpolate_on_Dim_{0}", dim.ind());
        return rewriter
                .create<VPU::InterpolateOp>(newLoc, inputVal, origOp.getSizes(), origOp.getScales(), origOp.getAxes(),
                                            newSizesAttr, newScalesAttr, origOp.getAxesAttrAttr(),
                                            origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(),
                                            origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr())
                .getOutput();
    };

    auto interpolateW = createSingleDimInterpOp(Dims4D::Act::W, origOp.getInput());
    auto interpolateH = createSingleDimInterpOp(Dims4D::Act::H, interpolateW);

    _log.nest().trace("[{0}] Split successful", getDebugName());
    rewriter.replaceOp(origOp, interpolateH);
    return mlir::success();
}

//
// SplitSEOpsPass
//

class SplitSEOpsPass final : public VPU::SplitSEOpsBase<SplitSEOpsPass> {
public:
    explicit SplitSEOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SplitSEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitInterpolate>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitSEOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitSEOpsPass(Logger log) {
    return std::make_unique<SplitSEOpsPass>(log);
}
