//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
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

Shape calcPadsEnd(mlir::ShapedType origType, int64_t channelAlignment) {
    const auto origShape = getShape(origType);

    auto extendedShape = origShape.toValues();
    extendedShape[IE::Dims4D::Act::C] = alignVal(origShape[IE::Dims4D::Act::C], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

//
// generalRewrite
//

//
// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input
//

mlir::LogicalResult generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator, Logger log) {
    auto* ctx = origOp->getContext();

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp);

    // TODO make input and output channel alignment - where is it specified?
    auto channelAlignement = iface.getChannelAlignment();

    const auto inputType = origOp->getOperand(0).getType().cast<mlir::ShapedType>();
    const auto outputType = origOp->getResult(0).getType().cast<mlir::ShapedType>();

    // TODO: Make better. If it is the first layer and < 16 input channels then aligning input channel to 16 is not
    // required.
    auto op = mlir::dyn_cast<IE::ConvolutionOp>(*origOp);
    const auto inputShape = getShape(op);
    const auto IC = inputShape[IE::Dims4D::Act::C];
    if (IC < 16)
        channelAlignement = 0;

    auto inPadsEnd = calcPadsEnd(inputType, channelAlignement);
    const auto outPadsEnd = calcPadsEnd(outputType, channelAlignement);

    // Overwrite for now for cm
    // inPadsEnd[IE::Dims4D::Act::C] = 0;
    // inPadsEnd[IE::Dims4D::Act::H] = 0;
    // inPadsEnd[IE::Dims4D::Act::W] = 0;
    // inPadsEnd[IE::Dims4D::Act::N] = 0;

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[IE::Dims4D::Act::C] == 0 && outPadsEnd[IE::Dims4D::Act::C] == 0) {
        return matchFailed(log, rewriter, origOp, "Both input and output channels are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[IE::Dims4D::Act::C] == 0) {
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput = rewriter.create<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(0), None, ShapeRef(inPadsEnd));
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, outPadsEnd[IE::Dims4D::Act::C]);

    if (outPadsEnd[IE::Dims4D::Act::C] == 0) {
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        const SmallVector<int64_t> offsets(outShape.size(), 0);

        auto subTensorOp =
                rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0),
                                             getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));

        rewriter.replaceOp(origOp, subTensorOp.result());
    }

    return mlir::success();
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("MaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[IE::Dims4D::Act::C] = outChanPadsEnd;

        const auto newOutputType = getPaddedType(origOp.getType(), outPadBefore, outPadAfter);

        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.kernel_size(),
                                              origOp.strides(), origOp.pads_begin(), origOp.pads_end(),
                                              origOp.rounding_type(), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto filterShape = getShape(origOp.filter());

        const auto newInputShape = getShape(expandedInput);
        const auto inChanPadEnd = newInputShape[IE::Dims4D::Act::C] - filterShape[IE::Dims4D::Filter::IC];

        mlir::Value paddedFilter;

        if (inChanPadEnd == 0 && outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            const SmallVector<int64_t> filterPadsBegin(filterShape.size(), 0);

            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[IE::Dims4D::Filter::OC] = outChanPadEnd;
            filterPadsEnd[IE::Dims4D::Filter::IC] = inChanPadEnd;

            const auto padValue = getFPAttr(getContext(), 0.0f);

            paddedFilter = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.filter(), nullptr, nullptr,
                                                            nullptr, getIntArrayAttr(getContext(), filterPadsBegin),
                                                            getIntArrayAttr(getContext(), filterPadsEnd.raw()),
                                                            padValue, IE::PadMode::CONSTANT);
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                const SmallVector<uint32_t> biasPadsBegin(biasShape.size(), 0);

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[IE::Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                const auto padValue = getFPAttr(getContext(), 0.0f);

                paddedBiases = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.bias(), nullptr, nullptr,
                                                                nullptr, getIntArrayAttr(getContext(), biasPadsBegin),
                                                                getIntArrayAttr(getContext(), biasPadsEnd.raw()),
                                                                padValue, IE::PadMode::CONSTANT);
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[IE::Dims4D::Act::C] = outChanPadEnd;

        const auto newOutputType = getPaddedType(origOp.getType(), outPadBefore, outPadAfter);

        return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), newOutputType, expandedInput, paddedFilter,
                                                  paddedBiases, origOp.strides(), origOp.pads_begin(),
                                                  origOp.pads_end(), origOp.dilations(), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// EltwiseAddRewriter
//

class EltwiseAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    EltwiseAddRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("EltwiseAddRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EltwiseAddRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AddOp layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput1, int64_t outChanPadEnd) -> mlir::Operation* {
        mlir::Value expandedInput2;
        if (outChanPadEnd == 0) {
            expandedInput2 = origOp.input2();
        } else {
            _log.trace("Expand second input tensor");

            const auto origShape = getShape(origOp.input2());
            const auto extendedShape = getShape(expandedInput1);
            VPUX_THROW_UNLESS(origShape.size() == extendedShape.size(), "Got non equal shapes in EltwiseAddRewriter");

            const auto padsEnd = calcPadsEnd(origShape, extendedShape);

            expandedInput2 = rewriter.create<IE::ExpandOp>(origOp->getLoc(), origOp.input2(), None, ShapeRef(padsEnd));
        }

        return rewriter.create<IE::AddOp>(origOp.getLoc(), expandedInput1, expandedInput2, origOp.auto_broadcast(),
                                          origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// GroupConvolutionRewriter
//

class GroupConvolutionRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("GroupConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolutionOp layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        const auto filterShape = getShape(origOp.filter());

        mlir::Value paddedFilter;

        if (outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            const SmallVector<int64_t> filterPadsBegin(filterShape.size(), 0);

            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[IE::Dims4D::Filter::OC] = outChanPadEnd;

            const auto padValue = getFPAttr(getContext(), 0.0);

            paddedFilter = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.filter(), nullptr, nullptr,
                                                            nullptr, getIntArrayAttr(getContext(), filterPadsBegin),
                                                            getIntArrayAttr(getContext(), filterPadsEnd.raw()),
                                                            padValue, IE::PadMode::CONSTANT);
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                const SmallVector<uint32_t> biasPadsBegin(biasShape.size(), 0);

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[IE::Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                const auto padValue = getFPAttr(getContext(), 0.0);

                paddedBiases = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.bias(), nullptr, nullptr,
                                                                nullptr, getIntArrayAttr(getContext(), biasPadsBegin),
                                                                getIntArrayAttr(getContext(), biasPadsEnd.raw()),
                                                                padValue, IE::PadMode::CONSTANT);
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[IE::Dims4D::Act::C] = outChanPadEnd;

        const auto newOutputType = getPaddedType(origOp.getType(), outPadBefore, outPadAfter);
        const auto newConvOutShape = getShape(newOutputType);

        return rewriter.create<IE::GroupConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, paddedBiases, origOp.strides(),
                origOp.pads_begin(), origOp.pads_end(), origOp.dilations(),
                getIntAttr(getContext(), newConvOutShape[IE::Dims4D::Act::C]), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// ExpandActivationChannelsPass
//

class ExpandActivationChannelsPass final : public IE::ExpandActivationChannelsBase<ExpandActivationChannelsPass> {
public:
    explicit ExpandActivationChannelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandActivationChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegal = [&](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            return iface.verifyChannels().succeeded();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(isLegal);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::PadOp, IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MaxPoolRewriter>(&ctx, _log);
    patterns.insert<ConvolutionRewriter>(&ctx, _log);
    patterns.insert<EltwiseAddRewriter>(&ctx, _log);
    patterns.insert<GroupConvolutionRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createExpandActivationChannelsPass(Logger log) {
    return std::make_unique<ExpandActivationChannelsPass>(log);
}
