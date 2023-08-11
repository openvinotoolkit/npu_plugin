//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvToNCE
//

class ConvToNCE final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("ConvToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inOrder != DimsOrder::NCHW && inOrder != DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, origOp, "Operation at '{0}' has unsupported input layout '{1}'",
                           origOp->getLoc(), inOrder);
    }
    if (!VPU::NCEConvolutionOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.contentAttr();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Generate activation window
    mlir::IntegerAttr activationWindowChannelLength;
    mlir::Value activationWindow = nullptr;
    bool isCMajorConv = inOrder == DimsOrder::NCHW;

    if (isCMajorConv) {
        const auto kernelSize = Shape{KY, KX};
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
        const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();

        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                        kernelStrides[Dims4D::Strides::X],
                                                                        origInputType.getElementType(), IC);

        const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                    kernelStrides[Dims4D::Strides::X],
                                                                    origInputType.getElementType(), IC);

        activationWindowChannelLength = getIntAttr(getContext(), bitPatternSize);
        activationWindow = VPU::createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity);
    }

    auto alignedFilter = VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), origOp.filter(), isCMajorConv);

    // Generate weights table
    const auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(origOp.input(), origOp.output(), origOp.post_opAttr(),
                                                                  origOp.getLoc(), origOp.getContext(), _arch);
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.input(), origOp.output(), alignedFilter, bias, OC,
                                                             ppeTaskAttr, _arch, origOp.post_opAttr());
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto instructionListTableVec =
            VPU::createInstructionListTableData(origOp.output(), origOp.post_opAttr(), _arch);
    const auto instructionListTable =
            VPU::createInstructionListTableTensor(rewriter, origOp->getLoc(), instructionListTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.input(), alignedFilter, weightsTable, activationWindow,
            instructionListTable, origOp.stridesAttr(), padAttr, ppeTaskAttr, rawFilterShape,
            activationWindowChannelLength, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// DepthConvToNCE
//

class DepthConvToNCE final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DepthConvToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEDepthConvolutionOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    // Get dimensions
    const auto filter = origOp.filter();
    const auto filterShape = getShape(filter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Generate activation window
    const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto origInputShape = origInputType.getShape();
    const auto IC = origInputShape[Dims4D::Act::C];

    const auto kernelSize = Shape{KY, KX};
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindowChannelLength = getIntAttr(getContext(), bitPatternSize);

    const auto fakeSparsity =
            VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                              kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindow = VPU::createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity);

    Const::ContentAttr bias;
    if (origOp.bias() != nullptr) {
        auto biasConstOp = origOp.bias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.contentAttr();
    }

    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(origOp.input(), origOp.output(), origOp.post_opAttr(),
                                                            origOp.getLoc(), origOp.getContext(), _arch);
    auto weightsTableVec = VPU::createWeightsTableData(origOp.input(), origOp.output(), alignedFilter, bias, OC,
                                                       ppeTaskAttr, _arch, origOp.post_opAttr());
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto instructionListTableVec =
            VPU::createInstructionListTableData(origOp.output(), origOp.post_opAttr(), _arch);
    const auto instructionListTable =
            VPU::createInstructionListTableTensor(rewriter, origOp->getLoc(), instructionListTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.input(), alignedFilter, weightsTable, activationWindow,
            instructionListTable, origOp.stridesAttr(), padAttr, ppeTaskAttr, rawFilterShape,
            activationWindowChannelLength, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

//
// MaxPoolToNCE
//

class MaxPoolToNCE final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolToNCE(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::NCEMaxPoolOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    // Get dimensions
    const auto origInputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = origInputType.getShape();

    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(origOp.kernel_size()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));

    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);

    // Generate activation window
    const auto fakeSparsity =
            VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::POOL, kernelSize,
                                              kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindow = VPU::createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity);
    const auto activationWindowChannelLength = getIntAttr(getContext(), static_cast<uint32_t>(bitPatternSize));

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(origOp.input(), origOp.output(), origOp.post_opAttr(),
                                                            origOp.getLoc(), origOp.getContext(), _arch);
    auto weightsTableVec = VPU::createWeightsTableData(origOp.input(), origOp.output(), nullptr, nullptr, IC,
                                                       ppeTaskAttr, _arch, origOp.post_opAttr());
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.pads_begin(), origOp.pads_end()));

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(origOp->getLoc(), origOp.getType(), origOp.input(), weightsTable,
                                                    activationWindow, origOp.kernel_sizeAttr(), origOp.stridesAttr(),
                                                    padAttr, ppeTaskAttr, activationWindowChannelLength,
                                                    /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.output());
    return mlir::success();
}

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
// EltwiseToNCE
//

template <class ConcreteOp>
class EltwiseToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::EltwiseType opType, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::EltwiseType _opType;
    VPU::ArchKind _arch;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", this->getDebugName(), msg.str());
    };

    // Multiply scales can be "compress" as one. You can make the same scale for each input.
    // in1 * in1_scale * in2 *in2_scale => in1 * in2 * (in1_scale * in2_scale).
    const bool allowDifferentScales =
            supportsPerInputEltwiseScale(_arch) ? true : _opType == VPU::EltwiseType::MULTIPLY;
    const bool allowDifferentZp = true;

    if (!VPU::NCEEltwiseOp::isSupported(origOp, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                        /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    auto ppeTaskAttr = VPU::getNCEEltwisePPETaskAttr(origOp.input1().getType(), origOp.input2().getType(),
                                                     origOp.output().getType(), origOp.post_opAttr(), origOp.getLoc(),
                                                     _opType, origOp.getContext(), _arch);

    auto nceOp =
            rewriter.create<VPU::NCEEltwiseOp>(origOp->getLoc(), origOp.getType(), origOp.input1(), origOp.input2(),
                                               VPU::EltwiseTypeAttr::get(this->getContext(), _opType), ppeTaskAttr,
                                               /*multi_cluster_strategyAttr=*/nullptr,
                                               /*is_inplace=*/nullptr);
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
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
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
    if (arch == VPU::ArchKind::VPUX37XX) {
        patterns.add<AveragePoolToNCE>(&ctx, arch, _log);
        patterns.add<PermuteQuantizeToNCE>(&ctx, _log);
    }
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, arch, _log);
    if (arch != VPU::ArchKind::VPUX37XX) {
        patterns.add<EltwiseToNCE<IE::MultiplyOp>>(&ctx, VPU::EltwiseType::MULTIPLY, arch, _log);
        patterns.add<EltwiseToNCE<IE::SubtractOp>>(&ctx, VPU::EltwiseType::SUBTRACT, arch, _log);
        patterns.add<EltwiseToNCE<IE::AndOp>>(&ctx, VPU::EltwiseType::AND, arch, _log);
    }

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}
