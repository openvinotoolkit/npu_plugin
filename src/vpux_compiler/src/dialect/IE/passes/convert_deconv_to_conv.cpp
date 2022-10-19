//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
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
    const auto contentAttr = constOp.contentAttr();
    const auto content = contentAttr.convertElemType(elemType).transpose(dimsOrder);
    return rewriter.create<Const::DeclareOp>(constOp.getLoc(), newConstType, content);
}

// Checks whether the Deconvolution filter is a constant or a FakeQuantize with a constant input
mlir::FailureOr<Const::DeclareOp> getConstFilter(IE::DeconvolutionOp deconv) {
    if (auto filterFq = deconv.filter().getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto filterConst = filterFq.input().getDefiningOp<Const::DeclareOp>()) {
            return filterConst;
        }
    } else if (auto filterConst = deconv.filter().getDefiningOp<Const::DeclareOp>()) {
        return filterConst;
    }
    return mlir::failure();
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

    VPUX_THROW_UNLESS(padsBeginVector == padsEndVector, "Supported only symmetrical paddings");

    const auto featureShape = getShape(origOp.feature());
    VPUX_THROW_UNLESS(featureShape.size() == 4, "Only 2D deconvolution is supported");

    const auto outputShape = getShape(origOp.output());
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Only 2D deconvolution is supported");

    auto filterShape = getShape(origOp.filter()).toValues();
    VPUX_THROW_UNLESS(filterShape.size() == 4, "Only 2D deconvolution is supported");

    auto padLX = filterShape[Dims4D::Filter::KX] - 1 - padsBeginVector[Dims4D::PadsBegin::Left];
    auto padRY = filterShape[Dims4D::Filter::KY] - 1 - padsEndVector[Dims4D::PadsEnd::Bottom];

    VPUX_THROW_WHEN((padLX < 0) || (filterShape[Dims4D::Filter::KX] - 1 - padsEndVector[Dims4D::PadsEnd::Right] < 0) ||
                            (filterShape[Dims4D::Filter::KY] - 1 - padsBeginVector[Dims4D::PadsBegin::Top] < 0) ||
                            (padRY < 0),
                    "Upsampling layer does not support negative paddings");

    auto upsamplingFactor = getIntArrayAttr(getContext(), SmallVector<int64_t>{stridesVector[Dims4D::Strides::X],
                                                                               stridesVector[Dims4D::Strides::Y], 1});
    auto padL = getIntArrayAttr(
            getContext(),
            SmallVector<int64_t>{padLX, filterShape[Dims4D::Filter::KY] - 1 - padsEndVector[Dims4D::PadsEnd::Right],
                                 0});
    auto padR = getIntArrayAttr(getContext(), SmallVector<int64_t>{filterShape[Dims4D::Filter::KX] - 1 -
                                                                           padsBeginVector[Dims4D::PadsBegin::Top],
                                                                   padRY, 0});
    auto newUpsamplingOp =
            rewriter.create<IE::UpsamplingOp>(origOp->getLoc(), origOp.feature(), upsamplingFactor, padL, padR);

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    const auto elemType = origOp.feature().getType().cast<vpux::NDTypeInterface>().getElementType();

    auto dwConvFilter = getConstFilter(origOp).getValue();
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

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            origOp, newUpsamplingOp.output(),
            fakeQuantizeOp != nullptr ? newFakeQuantizeOp.output() : reshapedFilter.output(), nullptr, strides,
            padsBegin, padsEnd, dilations, nullptr);

    _log.trace("Replaced with 'IE::Convolution' (2D)");

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
        // Check filter
        _log.trace("Got '{0}' at '{1}'", deconv->getName(), deconv->getLoc());
        if (mlir::failed(getConstFilter(deconv))) {
            _log.nest(1).trace("Deconv cannot be converted. Filter must be constant");
            return true;
        }

        // Check input shape
        const auto inputShape = getShape(deconv.feature());
        return inputShape.size() != 4;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::DeconvolutionOp>(isLegalDeconvOp);
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DeconvolutionConversion>(&ctx, _log);

    auto func = getFunction();
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
