//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/VPU37XX/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"

#include "vpux/compiler/VPU37XX/conversion.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// ConvToNCE
//

mlir::LogicalResult arch37xx::ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    const bool isCompressConvSupported = VPU::NCECompressConvolutionOp::isSupported(origOp, logCb,
                                                                                    /*checkLayout=*/true,
                                                                                    /*checkChannelAlignment=*/true);

    const auto filterShape = getShape(origOp.getFilter());
    auto OC = filterShape[Dims4D::Filter::OC];
    auto weightsConstValue = origOp.getFilter();
    auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);
    mlir::IntegerAttr cmSpPatternAttr;
    if (isCompressConvSupported) {
        auto weightsConstOp = weightsConstValue.getDefiningOp<Const::DeclareOp>();
        auto weightsContentAttr = weightsConstOp.getContentAttr();
        const auto origChannelVal =
                weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape()[Dims4D::Filter::IC];
        const auto outputChannels = origOp.getOutput().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto origShape = Shape(
                {outputChannels, origChannelVal, filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]});
        if (origShape[Dims4D::Filter::IC] != filterShape[Dims4D::Filter::IC]) {
            const auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
            auto newContentAttr = weightsConstOp.getContentAttr().subview(Shape(currentOffset), origShape);
            auto newConstType = weightsConstOp.getType().cast<NDTypeInterface>().changeShape(origShape);
            auto newWeightsConstOp =
                    rewriter.create<Const::DeclareOp>(weightsConstOp.getLoc(), newConstType, newContentAttr);
            weightsConstValue = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(newWeightsConstOp.getOutput());
            weightsConstOp.replaceAllUsesWith(newWeightsConstOp.getOperation());
        }
        rawFilterShape = getIntArrayAttr(rewriter, origShape);
        const int64_t cmSpPattern = (static_cast<int64_t>(1) << origChannelVal) - 1;
        cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);
    }

    auto alignedFilter =
            VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), weightsConstValue, /*isCMajorConv=*/false);

    // Generate weights table
    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }
    const auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedFilter, bias,
                                                             OC, ppeTaskAttr, _arch, origOp.getPostOpAttr());
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    if (isCompressConvSupported) {
        rewriter.replaceOpWithNewOp<VPU::NCECompressConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable, origOp.getStridesAttr(),
                padAttr, ppeTaskAttr, rawFilterShape,
                /*multi_cluster_strategyAttr=*/nullptr, cmSpPatternAttr);
    } else {
        rewriter.replaceOpWithNewOp<VPU::NCEConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
                /*activationWindow=*/nullptr, /*instructionListTable=*/nullptr, origOp.getStridesAttr(), padAttr,
                ppeTaskAttr, rawFilterShape,
                /*activation_window_channel_length=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr);
    };

    return mlir::success();
}

//
// DepthConvToNCE
//

mlir::LogicalResult arch37xx::DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Get dimensions
    const auto filter = origOp.getFilter();
    const auto filterShape = getShape(filter);
    const auto OC = filterShape[Dims4D::Filter::OC];

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }

    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);
    auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedFilter, bias, OC,
                                                       ppeTaskAttr, _arch, origOp.getPostOpAttr());
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
            /*activationWindow=*/nullptr, /*instructionListTable=*/nullptr, origOp.getStridesAttr(), padAttr,
            ppeTaskAttr, rawFilterShape,
            /*activation_window_channel_length=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// MaxPoolToNCE
//

mlir::LogicalResult arch37xx::MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), /*weightsTable=*/nullptr,
            /*activationWindow=*/nullptr, origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr, ppeTaskAttr,
            /*activation_window_channel_length=*/nullptr,
            /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

namespace {

//
// AveragePoolToNCE
//

class AveragePoolToNCE final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult AveragePoolToNCE::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ppeTaskAttr = VPU::getNCEAveragePoolPPETaskAttr(origOp.getInput().getType(), origOp.getKernelSizeAttr(),
                                                         origOp.getOutput().getType(), origOp.getPostOpAttr(),
                                                         origOp.getLoc(), origOp.getContext(), _arch);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    auto nceOp = rewriter.create<VPU::NCEAveragePoolOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(),
                                                        origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr,
                                                        ppeTaskAttr, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// PermuteQuantizeToNCE
//

class PermuteQuantizeToNCE final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    PermuteQuantizeToNCE(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::vector<int64_t> calculatePermuteQuantizeShape(const ShapeRef shape) {
    const int64_t tensorSizeY = shape[Dims4D::Act::C];
    const int64_t tensorSizeZ = shape[Dims4D::Act::W];
    const int64_t tensorSizeX = shape[Dims4D::Act::H];
    std::vector<int64_t> targetShape = {1, tensorSizeZ, tensorSizeY, tensorSizeX};
    return targetShape;
}

mlir::Value reshapeInput(const mlir::Location& loc, const mlir::Value input, mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();

    const auto inputShape = getShape(input);
    const auto targetShape = calculatePermuteQuantizeShape(inputShape);

    const auto inType = input.getType().cast<vpux::NDTypeInterface>();
    const auto reshapedInType = inType.changeShape(ShapeRef(targetShape));
    const auto reshapedInLoc = appendLoc(loc, " reshape_input_for_permuteQuantize: CHW -> WCH");
    auto reshapedIn = rewriter.create<VPU::ReshapeOp>(reshapedInLoc, reshapedInType, input, nullptr, false,
                                                      getIntArrayAttr(ctx, ShapeRef(targetShape)));

    return reshapedIn.getOutput();
}

mlir::Value reshapeOutput(const mlir::Location& loc, const mlir::Value output, const ShapeRef targetShape,
                          mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    const auto outType = output.getType().cast<vpux::NDTypeInterface>();
    const auto reshapedOutType = outType.changeShape(targetShape);
    const auto reshapedOutLoc = appendLoc(loc, " reshape output for permute quantize");
    SmallVector<SmallVector<int64_t>> reassociationMap(targetShape.size());
    for (size_t dimIdx = 0; dimIdx < targetShape.size(); dimIdx++) {
        reassociationMap[dimIdx].push_back(dimIdx);
    }
    auto reshapedOut = rewriter.create<IE::AffineReshapeOp>(reshapedOutLoc, reshapedOutType, output,
                                                            getIntArrayOfArray(ctx, reassociationMap),
                                                            getIntArrayAttr(ctx, targetShape));

    return reshapedOut.getOutput();
}

mlir::Value createPermuteQuantize(IE::PermuteQuantizeOp origOp, const mlir::Value reshapedIn,
                                  mlir::PatternRewriter& rewriter) {
    const auto outputShape = getShape(origOp.getOutput());
    const auto nceOutputShape = calculatePermuteQuantizeShape(outputShape);

    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto nceOutOpType = outType.changeShape(ShapeRef(nceOutputShape));
    const auto oduOrder = DimsOrder::NWCH;
    auto newAddOutType = nceOutOpType.changeDimsOrder(oduOrder);

    // Since channels become tensorOutSizeY now, top and bottom pads must be updated accordingly.
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr());
    const int64_t padLeft = 0;
    const int64_t padRight = 0;
    const int64_t padTop = padsBegin[Dims4D::Act::C.ind()];
    const int64_t padBottom = padsEnd[Dims4D::Act::C.ind()];
    const auto padAttr = VPU::getPaddingAttr(origOp.getContext(), PadInfo(padLeft, padRight, padTop, padBottom));

    // Despite the fact that IE.PermuteQuantize gets NCHW layout, DPU assumes NHWC input, but ignores strides.
    // Let's keep it transparent and explicitly cast NCHW PermuteQuantize input to NHWC DPU input.
    const auto targetInOrder = DimsOrder::NHWC;
    const auto orderInAttr = mlir::AffineMapAttr::get(targetInOrder.toAffineMap(origOp.getContext()));
    const auto inLayoutCastLoc = appendLoc(origOp.getLoc(), " input cast to NHWC");
    auto inLayoutCast = rewriter.create<VPU::LayoutCastOp>(inLayoutCastLoc, reshapedIn, orderInAttr);

    const auto oduOrderAttr = mlir::AffineMapAttr::get(oduOrder.toAffineMap(origOp.getContext()));
    const auto dstElemAttr = mlir::TypeAttr::get(newAddOutType.getElementType());
    auto nceOp = rewriter.create<VPU::NCEPermuteQuantizeOp>(origOp->getLoc(), newAddOutType, inLayoutCast.getOutput(),
                                                            padAttr, dstElemAttr, oduOrderAttr, nullptr,
                                                            /*multi_cluster_strategyAttr=*/nullptr);

    const auto targetOutOrder = DimsOrder::fromValue(origOp.getOutput());
    const auto orderOutAttr = mlir::AffineMapAttr::get(targetOutOrder.toAffineMap(origOp.getContext()));
    const auto outLayoutCastLoc = appendLoc(origOp.getLoc(), " output cast to NHWC");
    auto outLayoutCast = rewriter.create<VPU::LayoutCastOp>(outLayoutCastLoc, nceOp.getOutput(), orderOutAttr);
    if (nceOutOpType.getElementType().isF16() || nceOutOpType.getElementType().isF32()) {
        return outLayoutCast.getOutput();
    }

    const auto outLayoutCastType = outLayoutCast.getOutput().getType().cast<NDTypeInterface>();
    if (outLayoutCastType.getElementType() != nceOutOpType.getElementType()) {
        const auto quantCastLoc = appendLoc(origOp.getLoc(), " quant cast");
        auto quantCast = rewriter.create<IE::QuantizeCastOp>(quantCastLoc, outLayoutCast.getOutput(),
                                                             nceOutOpType.getElementType());
        return quantCast.getOutput();
    }

    return outLayoutCast.getOutput();
}

mlir::LogicalResult PermuteQuantizeToNCE::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());
    // PermuteQuantize on DPU takes NCHW input reshaped to 1x32xCxW, where W is the remainder of shape size / (32 * C)
    // NCEPermuteQuantize is executed via element-wise add. The input is added to itself and then divided by 2 in PPE.
    // NCEPermuteQuantize performs ODU permutation to YZX in super-dense mode, which results in NHWC output.
    auto reshapedIn = reshapeInput(origOp->getLoc(), origOp.getInput(), rewriter);
    auto nceOp = createPermuteQuantize(origOp, reshapedIn, rewriter);
    auto reshapedOut = reshapeOutput(origOp->getLoc(), nceOp, getShape(origOp.getOutput()), rewriter);
    rewriter.replaceOp(origOp, reshapedOut);
    return mlir::success();
}

//
// PermuteQuantizeToNCEPermute
//

class PermuteQuantizeToNCEPermute final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    PermuteQuantizeToNCEPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PermuteQuantizeToNCEPermute::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto expandedChannels = outType.getShape()[Dims4D::Act::C];
    const auto dstElemAttr = mlir::TypeAttr::get(outType.getElementType());

    auto nceOp = rewriter.create<VPU::NCEPermuteOp>(origOp->getLoc(), outType, origOp.getInput(),
                                                    getIntAttr(getContext(), expandedChannels), dstElemAttr,
                                                    origOp.getDstOrderAttr(),
                                                    /*ppeAttr=*/nullptr,
                                                    /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());

    return mlir::success();
}

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public arch37xx::ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(bool useNCEPermute, Logger log): _useNCEPermute(useNCEPermute), _log(log) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _useNCEPermute = false;
    Logger _log;
};

mlir::LogicalResult ConvertIEToVPUNCEPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (useNCEPermute.hasValue()) {
        _useNCEPermute = useNCEPermute.getValue();
    }

    return mlir::success();
}

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<vpux::IE::IEDialect>();
    target.addLegalDialect<vpux::VPU::VPUDialect>();
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        return !VPU::NCEConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true) &&
               !VPU::NCECompressConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                           /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        return !VPU::NCEDepthConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                        /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        return !VPU::NCEMaxPoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        return !VPU::NCEAveragePoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                   /*checkChannelAlignment=*/true);
    });

    if (!_useNCEPermute) {
        target.addDynamicallyLegalOp<IE::PermuteQuantizeOp>([&](IE::PermuteQuantizeOp op) {
            return !VPU::NCEPermuteQuantizeOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                           /*checkChannelAlignment=*/true);
        });
    }

    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) {
        const bool allowDifferentScales = true;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<arch37xx::ConvToNCE>(&ctx, arch, _log);
    patterns.add<arch37xx::DepthConvToNCE>(&ctx, arch, _log);
    patterns.add<arch37xx::MaxPoolToNCE>(&ctx, arch, _log);
    patterns.add<AveragePoolToNCE>(&ctx, arch, _log);
    if (!_useNCEPermute) {
        patterns.add<PermuteQuantizeToNCE>(&ctx, _log);
    }
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, arch, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    // TODO(E#81775): this can be integrated into the pattern set above instead of PermuteQuantizeToNCE
    if (_useNCEPermute) {
        mlir::ConversionTarget permuteTarget(ctx);
        permuteTarget.addDynamicallyLegalOp<IE::PermuteQuantizeOp>([&](IE::PermuteQuantizeOp op) {
            return !VPU::NCEPermuteOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkAlignment=*/true);
        });
        permuteTarget.addLegalOp<VPU::NCEPermuteOp>();
        mlir::RewritePatternSet ncePermutePattern(&ctx);
        ncePermutePattern.add<PermuteQuantizeToNCEPermute>(&ctx, _log);
        if (mlir::failed(mlir::applyPartialConversion(func, permuteTarget, std::move(ncePermutePattern)))) {
            signalPassFailure();
        }
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::arch37xx::createConvertIEToVPUNCEPass(bool useNCEPermute, Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(useNCEPermute, log);
}
