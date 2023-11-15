//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/VPU37XX/conversion.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

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

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEAveragePoolOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    auto ppeTaskAttr = VPU::getNCEAveragePoolPPETaskAttr(origOp.input().getType(), origOp.kernel_sizeAttr(),
                                                         origOp.output().getType(), origOp.post_opAttr(),
                                                         origOp.getLoc(), origOp.getContext(), _arch);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));

    auto nceOp = rewriter.create<VPU::NCEAveragePoolOp>(origOp->getLoc(), origOp.getType(), origOp.input(),
                                                        origOp.kernel_sizeAttr(), origOp.stridesAttr(), padAttr,
                                                        ppeTaskAttr, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());
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
    const std::vector<int64_t> targetShape = {1, tensorSizeZ, tensorSizeY, tensorSizeX};
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

    return reshapedIn.output();
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

    return reshapedOut.output();
}

mlir::Value createPermuteQuantize(IE::PermuteQuantizeOp origOp, const mlir::Value reshapedIn,
                                  mlir::PatternRewriter& rewriter) {
    const auto outputShape = getShape(origOp.output());
    const auto nceOutputShape = calculatePermuteQuantizeShape(outputShape);

    auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto nceOutOpType = outType.changeShape(ShapeRef(nceOutputShape));
    const auto oduOrder = DimsOrder::NWCH;
    auto newAddOutType = nceOutOpType.changeDimsOrder(oduOrder);

    // Since channels become tensorOutSizeY now, top and bottom pads must be updated accordingly.
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_beginAttr());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_endAttr());
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
    auto nceOp = rewriter.create<VPU::NCEPermuteQuantizeOp>(origOp->getLoc(), newAddOutType, inLayoutCast.output(),
                                                            padAttr, dstElemAttr, oduOrderAttr, nullptr,
                                                            /*multi_cluster_strategyAttr=*/nullptr);

    const auto targetOutOrder = DimsOrder::fromValue(origOp.output());
    const auto orderOutAttr = mlir::AffineMapAttr::get(targetOutOrder.toAffineMap(origOp.getContext()));
    const auto outLayoutCastLoc = appendLoc(origOp.getLoc(), " output cast to NHWC");
    auto outLayoutCast = rewriter.create<VPU::LayoutCastOp>(outLayoutCastLoc, nceOp.output(), orderOutAttr);
    if (nceOutOpType.getElementType().isF16() || nceOutOpType.getElementType().isF32()) {
        return outLayoutCast.output();
    }

    const auto quantCastLoc = appendLoc(origOp.getLoc(), " quant cast");
    auto quantCast =
            rewriter.create<IE::QuantizeCastOp>(quantCastLoc, outLayoutCast.output(), nceOutOpType.getElementType());

    return quantCast.output();
}

mlir::LogicalResult PermuteQuantizeToNCE::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };
    if (!VPU::NCEPermuteQuantizeOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    // PermuteQuantize on DPU takes NCHW input reshaped to 1x32xCxW, where W is the remainder of shape size / (32 * C)
    // NCEPermuteQuantize is executed via element-wise add. The input is added to itself and then divided by 2 in PPE.
    // NCEPermuteQuantize performs ODU permutation to YZX in super-dense mode, which results in NHWC output.
    auto reshapedIn = reshapeInput(origOp->getLoc(), origOp.input(), rewriter);
    auto nceOp = createPermuteQuantize(origOp, reshapedIn, rewriter);
    auto reshapedOut = reshapeOutput(origOp->getLoc(), nceOp, getShape(origOp.output()), rewriter);
    rewriter.replaceOp(origOp, reshapedOut);
    return mlir::success();
}

//
// ConvToNCECompressConv
//

class ConvToNCECompressConv final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvToNCECompressConv(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("ConvToNCECompressConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult ConvToNCECompressConv::matchAndRewrite(IE::ConvolutionOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto inputType = origOp.input().getType();
    const auto inputShape = inputType.cast<vpux::NDTypeInterface>().getShape();
    if (inputShape[Dims4D::Act::C] != VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) {
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder != DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, origOp, "Operation at '{0}' has unsupported input layout '{1}'",
                           origOp->getLoc(), inOrder);
    }
    if (!VPU::NCECompressConvolutionOp::isSupported(origOp, logCb)) {
        return mlir::failure();
    }

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.getContentAttr();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    auto weightsConstOp = origOp.filter().getDefiningOp<Const::DeclareOp>();
    auto weightsContentAttr = weightsConstOp.getContentAttr();
    const auto origChannelVal =
            weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape()[Dims4D::Filter::IC];
    const auto outputChannels = origOp.output().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto origShape = Shape({outputChannels, origChannelVal, KY, KX});

    const auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    auto newWeightsConstValue = weightsConstOp.getOutput();
    if (origShape[Dims4D::Filter::KY] == filterShape[Dims4D::Filter::KY] &&
        origShape[Dims4D::Filter::KX] == filterShape[Dims4D::Filter::KX] &&
        origShape[Dims4D::Filter::IC] != filterShape[Dims4D::Filter::IC]) {
        auto newContentAttr = weightsConstOp.getContentAttr().subview(Shape(currentOffset), origShape);
        auto newConstType = weightsConstOp.getType().cast<NDTypeInterface>().changeShape(origShape);
        auto newWeightsConstOp =
                rewriter.create<Const::DeclareOp>(weightsConstOp.getLoc(), newConstType, newContentAttr);
        newWeightsConstValue = newWeightsConstOp.getOutput();
        weightsConstOp.replaceAllUsesWith(newWeightsConstOp.getOperation());
    }

    auto alignedFilter = VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), newWeightsConstValue,
                                                     /*isCMajorConv=*/false);

    const auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(origOp.input(), origOp.output(), origOp.post_opAttr(),
                                                                  origOp.getLoc(), origOp.getContext(), _arch);
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.input(), origOp.output(), alignedFilter, bias, OC,
                                                             ppeTaskAttr, _arch, origOp.post_opAttr());
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, origShape);

    const int64_t cmSpPattern = (static_cast<int64_t>(1) << origChannelVal) - 1;
    auto cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);

    auto nceOp = rewriter.create<VPU::NCECompressConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.input(), alignedFilter, weightsTable, origOp.stridesAttr(),
            padAttr, ppeTaskAttr, rawFilterShape,
            /*multi_cluster_strategyAttr=*/nullptr, cmSpPatternAttr);

    rewriter.replaceOp(origOp, nceOp.output());
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
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };
    if (!VPU::NCEPermuteOp::isSupported(origOp, logCb, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    auto outType = origOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto expandedChannels = outType.getShape()[Dims4D::Act::C];
    const auto dstElemAttr = mlir::TypeAttr::get(outType.getElementType());

    auto nceOp = rewriter.create<VPU::NCEPermuteOp>(origOp->getLoc(), outType, origOp.input(),
                                                    getIntAttr(getContext(), expandedChannels), dstElemAttr,
                                                    origOp.dst_orderAttr(),
                                                    /*ppeAttr=*/nullptr,
                                                    /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());

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

private:
    void safeRunOnFunc() final;

private:
    const bool _useNCEPermute = false;
    Logger _log;
};

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvToNCE>(&ctx, arch, _log);
    patterns.add<DepthConvToNCE>(&ctx, arch, _log);
    patterns.add<MaxPoolToNCE>(&ctx, arch, _log);
    patterns.add<AveragePoolToNCE>(&ctx, arch, _log);
    if (useNCEPermute) {
        patterns.add<PermuteQuantizeToNCEPermute>(&ctx, _log);
    } else {
        patterns.add<PermuteQuantizeToNCE>(&ctx, _log);
    }
    patterns.add<ConvToNCECompressConv>(&ctx, arch, _log);
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, arch, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::arch37xx::createConvertIEToVPUNCEPass(bool useNCEPermute, Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(useNCEPermute, log);
}
