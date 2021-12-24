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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

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

    auto padsBeginVector = parseIntArrayAttr<int32_t>(origOp.pads_begin());
    auto padsEndVector = parseIntArrayAttr<int32_t>(origOp.pads_end());
    auto stridesVector = parseIntArrayAttr<int32_t>(origOp.strides());

    VPUX_THROW_UNLESS(padsBeginVector == padsEndVector, "Supported only symmetrical paddings");

    const auto featureShape = origOp.feature().getType().cast<mlir::ShapedType>().getShape();
    auto featureShapeVector = to_small_vector(featureShape);
    VPUX_THROW_UNLESS(featureShapeVector.size() == 4, "Only 2D deconvolution is supported");

    const auto outputShape = origOp.output().getType().cast<mlir::ShapedType>().getShape();
    auto outputShapeVector = to_small_vector(outputShape);
    VPUX_THROW_UNLESS(outputShapeVector.size() == 4, "Only 2D deconvolution is supported");

    auto origOutputX =
            stridesVector[0] * (featureShapeVector[3] - 1) + stridesVector[0] - padsBeginVector[0] - padsEndVector[0];
    auto origOutputY =
            stridesVector[1] * (featureShapeVector[2] - 1) + stridesVector[1] - padsBeginVector[1] - padsEndVector[1];
    VPUX_THROW_UNLESS(origOutputX == outputShapeVector[3] && origOutputY == outputShapeVector[2],
                      "Unsupported output shape");

    const auto filterShape = origOp.filter().getType().cast<mlir::ShapedType>().getShape();
    auto filterShapeVector = to_small_vector(filterShape);
    VPUX_THROW_UNLESS(filterShapeVector.size() == 4, "Only 2D deconvolution is supported");

    VPUX_THROW_WHEN((filterShapeVector[3] - 1 - padsBeginVector[0] < 0) ||
                            (filterShapeVector[3] - 1 - padsEndVector[0] < 0) ||
                            (filterShapeVector[2] - 1 - padsBeginVector[1] < 0) ||
                            (filterShapeVector[2] - 1 - padsEndVector[1] < 0),
                    "Upsampling layer does not support negative paddings");

    auto upsamplingFactor = getIntArrayAttr(getContext(), SmallVector<int64_t>{stridesVector[0], stridesVector[1], 1});
    auto padL = getIntArrayAttr(getContext(), SmallVector<int64_t>{(filterShapeVector[3] - 1) - padsBeginVector[0],
                                                                   (filterShapeVector[2] - 1) - padsEndVector[0], 0});
    auto padR = getIntArrayAttr(getContext(), SmallVector<int64_t>{(filterShapeVector[3] - 1) - padsBeginVector[1],
                                                                   (filterShapeVector[2] - 1) - padsEndVector[1], 0});
    auto newUpsamplingOp =
            rewriter.create<IE::UpsamplingOp>(origOp->getLoc(), origOp.feature(), upsamplingFactor, padL, padR);

    auto strides = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(getContext(), SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(getContext(), SmallVector<int64_t>{1, 1});

    const auto elemType = origOp.feature().getType().cast<mlir::ShapedType>().getElementType();
    const auto dataStorageType = mlir::RankedTensorType::get(filterShapeVector, elemType);
    const auto dwConvFilterContent = origOp.filter().getDefiningOp<Const::DeclareOp>().content();

    // Weights reverse according to ngraph implementation
    // https://github.com/openvinotoolkit/openvino/blob/745c8933bc67f0eaf7996848f5188521fdf50d14/ngraph/core/reference/include/ngraph/runtime/reference/convolution_backprop_data.hpp#L268
    SmallVector<float16> reversed_vals(dwConvFilterContent.getValues<float16>());
    size_t spatialDims = filterShapeVector[3] * filterShapeVector[2];
    for (auto it = reversed_vals.begin(); it < reversed_vals.end(); it += spatialDims) {
        std::reverse(it, it + spatialDims);
    }

    SmallVector<float16> reshaped_vals(reversed_vals.size());
    for (auto c_in = 0; c_in < filterShapeVector[0]; c_in++) {
        for (auto c_out = 0; c_out < filterShapeVector[1]; c_out++) {
            auto inInd = c_out * spatialDims + c_in * spatialDims * filterShapeVector[1];
            auto outInd = c_in * spatialDims + c_out * spatialDims * filterShapeVector[0];

            std::copy(reversed_vals.begin() + inInd, reversed_vals.begin() + inInd + spatialDims,
                      reshaped_vals.begin() + outInd);
        }
    }

    auto C_IN = filterShapeVector[0];
    filterShapeVector[0] = filterShapeVector[1];
    filterShapeVector[1] = C_IN;

    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(reshaped_vals));
    auto reversedFilter =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, Const::ContentAttr::get(dataAttr));

    const auto newFilterShapeAtter = getIntArrayAttr(getContext(), filterShapeVector);
    auto reshapedFilter =
            rewriter.create<IE::ReshapeOp>(origOp.getLoc(), reversedFilter, nullptr, false, newFilterShapeAtter);

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origOp, newUpsamplingOp.output(), reshapedFilter.output(), nullptr,
                                                   strides, padsBegin, padsEnd, dilations, nullptr);

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

    const auto isLegalConvOp = [&](IE::DeconvolutionOp conv) {
        const auto inputShape = conv.feature().getType().cast<mlir::ShapedType>().getShape();
        return inputShape.size() != 4;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::DeconvolutionOp>(isLegalConvOp);
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();

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
