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
#include "vpux/compiler/core/layers.hpp"
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

Shape calcPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    extendedShape[Dims4D::Act::C] = alignVal(origShape[Dims4D::Act::C], channelAlignment);

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

    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    const auto inPadsEnd = calcPadsEnd(inputType, iface.getInputChannelAlignment());
    const auto outPadsEnd = calcPadsEnd(outputType, iface.getOutputChannelAlignment());

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[Dims4D::Act::C] == 0 && outPadsEnd[Dims4D::Act::C] == 0) {
        return matchFailed(log, rewriter, origOp, "Both input and output channels are already aligned");
    }

    // For Eltwise ops in case when input and output have diffenet element types (e.g. quantized input and float output)
    //    if (origOp->hasTrait<IE::EltwiseOp>() && (inPadsEnd[Dims4D::Act::C] != outPadsEnd[Dims4D::Act::C])) {
    //        auto maxPadsEnd = std::max<int64_t>(inPadsEnd[Dims4D::Act::C], outPadsEnd[Dims4D::Act::C]);
    //        inPadsEnd[Dims4D::Act::C] = maxPadsEnd;
    //        outPadsEnd[Dims4D::Act::C] = maxPadsEnd;
    //    }

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput =
                rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(0), None, ShapeRef(inPadsEnd));
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, outPadsEnd[Dims4D::Act::C]);

    if (outPadsEnd[Dims4D::Act::C] == 0) {
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
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

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
        const auto inChanPadEnd = newInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];

        mlir::Value paddedFilter;

        if (inChanPadEnd == 0 && outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;
            filterPadsEnd[Dims4D::Filter::IC] = inChanPadEnd;

            paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.filter(), None,
                                                               ShapeRef(filterPadsEnd));
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.bias(), None,
                                                                   ShapeRef(biasPadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), newOutputType, expandedInput, paddedFilter,
                                                  paddedBiases, origOp.strides(), origOp.pads_begin(),
                                                  origOp.pads_end(), origOp.dilations(), origOp.post_opAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, _log.nest());
}

//
// EltwiseRewriter
//

template <class ConcreteOp>
class EltwiseRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("EltwiseRewriter");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got eltwise layer at '{1}'", this->getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput1, int64_t outChanPadEnd) -> mlir::Operation* {
        mlir::Value expandedInput2;
        if (origOp.input1() == origOp.input2()) {
            expandedInput2 = expandedInput1;
        } else if (outChanPadEnd == 0) {
            expandedInput2 = origOp.input2();
        } else {
            _log.trace("Expand second input tensor");

            const auto origShape = getShape(origOp.input2());
            const auto extendedShape = getShape(expandedInput1);
            VPUX_THROW_UNLESS(origShape.size() == extendedShape.size(), "Got non equal shapes in EltwiseRewriter");

            const auto padsEnd = calcPadsEnd(origShape, extendedShape);

            expandedInput2 =
                    rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.input2(), None, ShapeRef(padsEnd));
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().template cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<ConcreteOp>(origOp.getLoc(), newOutputType, expandedInput1, expandedInput2,
                                           origOp.auto_broadcast(), origOp.post_opAttr());
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
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;

            paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.filter(), None,
                                                               ShapeRef(filterPadsEnd));
        }

        mlir::Value paddedBiases;

        if (origOp.bias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.bias();
            } else {
                const auto biasShape = getShape(origOp.bias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.bias(), None,
                                                                   ShapeRef(biasPadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);
        const auto newConvOutShape = newOutputType.getShape().toValues();

        return rewriter.create<IE::GroupConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, paddedBiases, origOp.strides(),
                origOp.pads_begin(), origOp.pads_end(), origOp.dilations(),
                getIntAttr(getContext(), newConvOutShape[Dims4D::Act::C]), origOp.post_opAttr());
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
    patterns.insert<EltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.insert<EltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);
    patterns.insert<EltwiseRewriter<IE::SubtractOp>>(&ctx, _log);
    patterns.insert<EltwiseRewriter<IE::AndOp>>(&ctx, _log);
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
