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

mlir::Value reshapeFQParams(mlir::Value input, DimsOrder dimsOrder, mlir::PatternRewriter& rewriter) {
    auto shape = getShape(input).toValues();
    auto constOp = input.getDefiningOp<Const::DeclareOp>();
    const auto elemType = constOp.getType().cast<vpux::NDTypeInterface>().getElementType();

    std::swap(shape[Dims4D::Filter::OC], shape[Dims4D::Filter::IC]);
    const auto newConstType =
            mlir::RankedTensorType::get(to_small_vector(shape), elemType).cast<vpux::NDTypeInterface>();
    const auto contentAttr = constOp.getContentAttr();
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

    auto padsOutput = Shape(parseIntArrayAttr<int64_t>(origOp.output_padding()));

    const auto featureShape = getShape(origOp.feature());
    VPUX_THROW_UNLESS(featureShape.size() == 4, "Only 2D deconvolution is supported");

    const auto outputShape = getShape(origOp.output());
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Only 2D deconvolution is supported");

    auto filterShape = getShape(origOp.filter()).toValues();
    VPUX_THROW_UNLESS(filterShape.size() == 4, "Only 2D deconvolution is supported");

    auto featureUpScale = IE::createUpsampling(rewriter, origOp, padsOutput, false);
    if (mlir::failed(featureUpScale)) {
        _log.nest().trace("Failed to create Upsampling for {0}", origOp->getLoc());
        return mlir::failure();
    }
    auto paddingOutput = featureUpScale.value();

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    const auto elemType = origOp.feature().getType().cast<vpux::NDTypeInterface>().getElementType();

    auto dwConvFilter = IE::getConstFilter(origOp).value();
    const auto filterContentAttr = dwConvFilter.getContentAttr();

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
                fakeQuantizeOp.getLoc(), reshapedFilter.getOutput(), newInputLow, newInputHigh, newOutputLow,
                newOutputHigh, fakeQuantizeOp.levels(), fakeQuantizeOp.auto_broadcast());
    }

    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp.output().user_begin()));

    const auto postOp = origOp.post_opAttr();

    if (padsOutput[Dims4D::PadsOutput::Y] > 0) {
        paddingOutput = createPadding(rewriter, origOp->getLoc(), paddingOutput, Dims4D::Act::H,
                                      padsOutput[Dims4D::PadsOutput::Y], outputFQ);
    }
    if (padsOutput[Dims4D::PadsOutput::X] > 0) {
        paddingOutput = createPadding(rewriter, origOp->getLoc(), paddingOutput, Dims4D::Act::W,
                                      padsOutput[Dims4D::PadsOutput::X], outputFQ);
    }

    auto resultOP = rewriter.create<IE::ConvolutionOp>(
                                    origOp.getLoc(), paddingOutput,
                                    fakeQuantizeOp != nullptr ? newFakeQuantizeOp.output() : reshapedFilter.getOutput(),
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
