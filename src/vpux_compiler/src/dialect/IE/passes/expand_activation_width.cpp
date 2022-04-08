//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape) {
    Shape padsEnd(origShape.size());

    for (auto i : irange(origShape.size())) {
        const auto d = Dim(i);
        padsEnd[d] = extendedShape[d] - origShape[d];
    }

    return padsEnd;
}

Shape calcOutPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    extendedShape[Dims4D::Act::W] = alignVal(origShape[Dims4D::Act::W], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

Shape calcInPadsEnd(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType, const ShapeRef outputPads,
                    const int64_t kernelX, const int64_t strideX) {
    const auto inputShape = inputType.getShape();
    const auto outputShape = outputType.getShape();
    const auto outputWidth = outputShape[Dims4D::Act::W] + outputPads[Dims4D::Act::W];

    auto extendedShape = inputShape.toValues();
    extendedShape[Dims4D::Act::W] = (outputWidth - 1) * strideX + kernelX;

    return calcPadsEnd(inputShape, extendedShape);
}

mlir::Operation* opCreator(mlir::Operation* origOp, vpux::NDTypeInterface ndType, ArrayRef<mlir::Value> expandedInputs,
                           int64_t outWidthPadEnd, mlir::PatternRewriter& rewriter) {
    const Shape outPadBefore(checked_cast<size_t>(ndType.getRank()), 0);

    Shape outPadAfter(checked_cast<size_t>(ndType.getRank()), 0);
    outPadAfter[Dims4D::Act::W] = outWidthPadEnd;

    const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

    auto* newNCEOp = rewriter.clone(*origOp);
    for (size_t inIdx = 0; inIdx < expandedInputs.size(); inIdx++) {
        newNCEOp->setOperand(checked_cast<unsigned>(inIdx), expandedInputs[inIdx]);
    }
    newNCEOp->getResult(0).setType(newOutputType);
    return newNCEOp;
}

//
// generalRewrite
//

mlir::LogicalResult generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, const int64_t kernelX,
                                   const int64_t strideX, const bool isEltwise, Logger log) {
    auto* ctx = origOp->getContext();

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp);

    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    const auto outPadsEnd = calcOutPadsEnd(outputType, iface.getOutputChannelAlignment());
    const auto inPadsEnd = calcInPadsEnd(inputType, outputType, outPadsEnd, kernelX, strideX);

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[Dims4D::Act::W] == 0 && outPadsEnd[Dims4D::Act::W] == 0) {
        return matchFailed(log, rewriter, origOp, "Both input and output width are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::W] == 0) {
        log.trace("Input width is already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput =
                rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(0), None, ShapeRef(inPadsEnd));
    }

    SmallVector<mlir::Value> paddedInputs = {paddedInput};
    // Check if element-wise operation has same value in both operands
    if (isEltwise) {
        if (origOp->getOperand(0) == origOp->getOperand(1)) {
            // Same input. Push it into the vector twice.
            paddedInputs.push_back(paddedInput);
        } else if (inPadsEnd[Dims4D::Act::W] == 0) {
            // No need to pad. Store the original value.
            paddedInputs.push_back(origOp->getOperand(1));
        } else {
            log.trace("Expand second input tensor");

            paddedInputs.push_back(rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(1), None,
                                                                       ShapeRef(inPadsEnd)));
        }
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(origOp, outputType, paddedInputs, outPadsEnd[Dims4D::Act::W], rewriter);

    if (outPadsEnd[Dims4D::Act::W] == 0) {
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        const SmallVector<int64_t> offsets(outShape.size(), 0);

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp->getResult(0).getType(), newOp->getResult(0),
                                                 getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));
    }

    return mlir::success();
}

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        this->setDebugName("ConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.Convolution at '{1}'", this->getDebugName(), origOp->getLoc());
    const auto filterShape = getShape(origOp.filter());
    const auto kernelX = filterShape[Dims4D::Filter::KX];
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto strideX = strides[Dims4D::Strides::X.ind()];
    return generalRewrite(origOp, rewriter, kernelX, strideX, false, _log);
}

//
// GroupConvolutionRewriter
//

class GroupConvolutionRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        this->setDebugName("GroupConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.GroupConvolution at '{1}'", this->getDebugName(), origOp->getLoc());
    const auto filterShape = getShape(origOp.filter());
    const auto kernelX = filterShape[Dims4D::Filter::KX];
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto strideX = strides[Dims4D::Strides::X.ind()];
    return generalRewrite(origOp, rewriter, kernelX, strideX, false, _log);
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        this->setDebugName("MaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.MaxPool at '{1}'", this->getDebugName(), origOp->getLoc());
    const auto kernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto kernelX = kernel[Dims4D::Kernel::X.ind()];
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto strideX = strides[Dims4D::Strides::X.ind()];
    return generalRewrite(origOp, rewriter, kernelX, strideX, false, _log);
}

//
// AvgPoolRewriter
//

class AvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        this->setDebugName("AvgPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.AvgPool at '{1}'", this->getDebugName(), origOp->getLoc());
    const auto kernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto kernelX = kernel[Dims4D::Kernel::X.ind()];
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto strideX = strides[Dims4D::Strides::X.ind()];
    return generalRewrite(origOp, rewriter, kernelX, strideX, false, _log);
}

//
// EltwiseAddRewriter
//

class EltwiseAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    EltwiseAddRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        this->setDebugName("EltwiseAddRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EltwiseAddRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.Add at '{1}'", this->getDebugName(), origOp->getLoc());
    const auto lhsType = origOp.input1().getType().cast<vpux::NDTypeInterface>();
    const auto rhsType = origOp.input2().getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(lhsType.getShape() == rhsType.getShape(), "Broadcast is not supported in EltwiseAddRewriter");

    return generalRewrite(origOp, rewriter, 1, 1, true, _log);
}

//
// ExpandActivationWidthPass
//

class ExpandActivationWidthPass final : public IE::ExpandActivationWidthBase<ExpandActivationWidthPass> {
public:
    explicit ExpandActivationWidthPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandActivationWidthPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    const auto arch = VPU::getArch(func->getParentOfType<mlir::ModuleOp>());
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) == 0) {
        _log.trace("ExpandActivationWidthPass is only applicable for VPUX37XX device.");
        return;
    }

    const auto isLegal = [&](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
            const auto outOrder = DimsOrder::fromValue(op->getResult(0));
            // With other configurations, width padding does not apply properly
            if (inOrder != DimsOrder::NHWC) {
                return true;
            }
            if (outOrder != DimsOrder::NCHW) {
                return true;
            }

            const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
            const auto outPadsEnd = calcOutPadsEnd(outputType, iface.getOutputChannelAlignment());

            return outPadsEnd[Dims4D::Act::W] == 0;
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegal);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegal);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(isLegal);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(isLegal);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegal);
    target.addLegalOp<Const::DeclareOp, IE::ExpandOp, IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionRewriter>(&ctx, _log);
    patterns.add<GroupConvolutionRewriter>(&ctx, _log);
    patterns.add<MaxPoolRewriter>(&ctx, _log);
    patterns.add<AvgPoolRewriter>(&ctx, _log);
    patterns.add<EltwiseAddRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createExpandActivationWidthPass(Logger log) {
    return std::make_unique<ExpandActivationWidthPass>(log);
}
