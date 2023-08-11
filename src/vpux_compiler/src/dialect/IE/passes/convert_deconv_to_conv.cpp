//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/deconvolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

mlir::Value createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.output().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(input));
    return rewriter
            .create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, input, fq.input_low(), fq.input_high(),
                                        fq.output_low(), fq.output_high(), fq.levels(), fq.auto_broadcast())
            ->getResult(0);
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, IE::DeconvolutionOp origOp, mlir::Value input, Dim axis,
                          int64_t nums, IE::FakeQuantizeOp inputFQ) {
    auto inputShape = getShape(input);
    auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
    offsets[axis.ind()] = inputShape[axis] - 1;
    sizes[axis.ind()] = 1;

    auto subSlice = rewriter.create<IE::SliceOp>(origOp->getLoc(), input, getIntArrayAttr(origOp.getContext(), offsets),
                                                 getIntArrayAttr(origOp.getContext(), sizes))
                            .result();
    if (inputFQ != nullptr) {
        subSlice = createFQ(rewriter, subSlice, inputFQ);
    }

    SmallVector<mlir::Value> subSlices;
    subSlices.push_back(input);
    subSlices.insert(subSlices.end(), nums, subSlice);
    auto concatOp = rewriter.create<IE::ConcatOp>(origOp->getLoc(), subSlices, axis).output();
    if (inputFQ != nullptr) {
        concatOp = createFQ(rewriter, concatOp, inputFQ);
    }

    return concatOp;
}

mlir::Value reshapeFQParams(mlir::Value input, DimsOrder dimsOrder, mlir::PatternRewriter& rewriter) {
    auto shape = getShape(input).toValues();
    auto constOp = input.getDefiningOp<Const::DeclareOp>();
    const auto elemType = constOp.getType().cast<vpux::NDTypeInterface>().getElementType();

    std::swap(shape[Dims4D::Filter::OC], shape[Dims4D::Filter::IC]);
    const auto newConstType =
            mlir::RankedTensorType::get(to_small_vector(shape), elemType).cast<vpux::NDTypeInterface>();
    const auto contentAttr = constOp.contentAttr();
    const auto content = contentAttr.convertElemType(elemType).transpose(dimsOrder);
    return rewriter.create<Const::DeclareOp>(constOp.getLoc(), newConstType, content);
}

//
// DeconvolutionConversion
//

class DeconvolutionConversion final : public mlir::OpRewritePattern<IE::DeconvolutionOp> {
public:
    DeconvolutionConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DeconvolutionOp>(ctx), _log(log) {
        setDebugName("DeconvolutionConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DeconvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DeconvolutionConversion::matchAndRewrite(IE::DeconvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Deconvolution Operation '{0}'", origOp->getLoc());

    const auto padsBeginVector = Shape(parseIntArrayAttr<int64_t>(origOp.pads_begin()));
    const auto padsEndVector = Shape(parseIntArrayAttr<int64_t>(origOp.pads_end()));
    const auto stridesVector = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
    auto padsOutput = Shape(parseIntArrayAttr<int64_t>(origOp.output_padding()));

    const auto featureShape = getShape(origOp.feature());
    VPUX_THROW_UNLESS(featureShape.size() == 4, "Only 2D deconvolution is supported");

    const auto outputShape = getShape(origOp.output());
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Only 2D deconvolution is supported");

    auto filterShape = getShape(origOp.filter()).toValues();
    VPUX_THROW_UNLESS(filterShape.size() == 4, "Only 2D deconvolution is supported");

    auto padL = filterShape[Dims4D::Filter::KX] - 1 - padsBeginVector[Dims4D::PadsBegin::Left];
    auto padR = filterShape[Dims4D::Filter::KX] - 1 - padsEndVector[Dims4D::PadsEnd::Right];
    auto padT = filterShape[Dims4D::Filter::KY] - 1 - padsBeginVector[Dims4D::PadsBegin::Top];
    auto padB = filterShape[Dims4D::Filter::KY] - 1 - padsEndVector[Dims4D::PadsEnd::Bottom];

    // Output_padding refers to copying convolutional input data. If the value of Output_padding is less than the value
    // of PadR&PadB, the copied data will be 0, so it can be merged with PadR&PadB.
    if ((padsOutput[Dims4D::PadsOutput::Y] > 0) && (padsOutput[Dims4D::PadsOutput::Y] <= padB)) {
        padB += padsOutput[Dims4D::PadsOutput::Y];
        padsOutput[Dims4D::PadsOutput::Y] = 0;
    }
    if ((padsOutput[Dims4D::PadsOutput::X] > 0) && (padsOutput[Dims4D::PadsOutput::X] <= padR)) {
        padR += padsOutput[Dims4D::PadsOutput::X];
        padsOutput[Dims4D::PadsOutput::X] = 0;
    }

    auto padChannelAttr = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padHeightAttr = getIntArrayAttr(getContext(), SmallVector<int64_t>{padT, padB});
    auto padWidthAttr = getIntArrayAttr(getContext(), SmallVector<int64_t>{padL, padR});
    auto padAttr = IE::UpsamplingPadAttr::get(padChannelAttr, padHeightAttr, padWidthAttr, getContext());

    VPUX_THROW_WHEN(((padL < 0) || (padR < 0) || (padT < 0) || (padB < 0)),
                    "Upsampling layer does not support negative paddings");

    auto upsamplingFactor = getIntArrayAttr(getContext(), SmallVector<int64_t>{stridesVector[Dims4D::Strides::X],
                                                                               stridesVector[Dims4D::Strides::Y], 1});

    auto paddingOutput =
            rewriter.create<IE::UpsamplingOp>(origOp->getLoc(), origOp.feature(), upsamplingFactor, padAttr).output();
    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    const auto elemType = origOp.feature().getType().cast<vpux::NDTypeInterface>().getElementType();

    auto dwConvFilter = IE::getConstFilter(origOp).getValue();
    const auto filterContentAttr = dwConvFilter.contentAttr();

    std::swap(filterShape[Dims4D::Filter::OC], filterShape[Dims4D::Filter::IC]);
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(filterShape), elemType);

    const auto content =
            filterContentAttr.reverse(Dims4D::Filter::IC).convertElemType(elemType).transpose(DimsOrder::IOYX);
    auto reshapedFilter = rewriter.create<Const::DeclareOp>(dwConvFilter.getLoc(), dataStorageType, content);
    auto fakeQuantizeOp = origOp.filter().getDefiningOp<IE::FakeQuantizeOp>();
    IE::FakeQuantizeOp newFakeQuantizeOp = nullptr;
    if (fakeQuantizeOp != nullptr) {
        auto newInputLow = reshapeFQParams(fakeQuantizeOp.input_low(), DimsOrder::IOYX, rewriter);
        auto newOutputLow = (fakeQuantizeOp.input_low() == fakeQuantizeOp.output_low())
                                    ? newInputLow
                                    : reshapeFQParams(fakeQuantizeOp.output_low(), DimsOrder::IOYX, rewriter);
        auto newInputHigh = reshapeFQParams(fakeQuantizeOp.input_high(), DimsOrder::IOYX, rewriter);
        auto newOutputHigh = (fakeQuantizeOp.input_high() == fakeQuantizeOp.output_high())
                                     ? newInputHigh
                                     : reshapeFQParams(fakeQuantizeOp.output_high(), DimsOrder::IOYX, rewriter);
        newFakeQuantizeOp = rewriter.create<IE::FakeQuantizeOp>(
                fakeQuantizeOp.getLoc(), reshapedFilter.output(), newInputLow, newInputHigh, newOutputLow,
                newOutputHigh, fakeQuantizeOp.levels(), fakeQuantizeOp.auto_broadcast());
    }

    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp.output().user_begin()));

    const auto postOp = origOp.post_opAttr();

    if (padsOutput[Dims4D::PadsOutput::Y] > 0) {
        paddingOutput = createPadding(rewriter, origOp, paddingOutput, Dims4D::Act::H,
                                      padsOutput[Dims4D::PadsOutput::Y], outputFQ);
    }
    if (padsOutput[Dims4D::PadsOutput::X] > 0) {
        paddingOutput = createPadding(rewriter, origOp, paddingOutput, Dims4D::Act::W,
                                      padsOutput[Dims4D::PadsOutput::X], outputFQ);
    }

    auto resultOP = rewriter.create<IE::ConvolutionOp>(
                                    origOp.getLoc(), paddingOutput,
                                    fakeQuantizeOp != nullptr ? newFakeQuantizeOp.output() : reshapedFilter.output(),
                                    nullptr, strides, padsBegin, padsEnd, dilations, postOp)
                            .output();

    rewriter.replaceOp(origOp, resultOP);

    _log.trace("Replaced DeConvolution at '{0}' with 'IE::Convolution' (2D)", origOp.getLoc());

    return mlir::success();
}

//
// ConvertDeconv2DToConv2DPass
//

class ConvertDeconv2DToConv2DPass final : public IE::ConvertDeconv2DToConv2DBase<ConvertDeconv2DToConv2DPass> {
public:
    explicit ConvertDeconv2DToConv2DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDeconv2DToConv2DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalDeconvOp = [&](IE::DeconvolutionOp deconv) {
        _log.trace("Got '{0}' at '{1}'", deconv->getName(), deconv->getLoc());
        if (mlir::failed(IE::canConvertDeconvToConv(deconv))) {
            _log.nest(1).trace("Deconv cannot be converted. Filter must be constant");
            return true;
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::DeconvolutionOp>(isLegalDeconvOp);
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DeconvolutionConversion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDeconv2DToConv2DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDeconv2DToConv2DPass(Logger log) {
    return std::make_unique<ConvertDeconv2DToConv2DPass>(log);
}
