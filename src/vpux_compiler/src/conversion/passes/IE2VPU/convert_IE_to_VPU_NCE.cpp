//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"

using namespace vpux;

//
// ConvToNCE
//

mlir::LogicalResult ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto inputType = origOp.input().getType();
    const auto inputShape = inputType.cast<vpux::NDTypeInterface>().getShape();
    const auto inOrder = DimsOrder::fromValue(origOp.input());
    if (inputShape[Dims4D::Act::C] == VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM &&
        inOrder != DimsOrder::NCHW) {
        return mlir::failure();
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

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

        bias = biasConstOp.getContentAttr();
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

        bias = biasConstOp.getContentAttr();
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
