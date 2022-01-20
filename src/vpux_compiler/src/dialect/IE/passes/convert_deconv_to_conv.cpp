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

#include "vpux/compiler/core/layers.hpp"
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
    const auto dataStorageType =
            mlir::RankedTensorType::get(to_small_vector(filterShape), elemType).cast<vpux::NDTypeInterface>();

    if (origOp.filter().getDefiningOp<Const::DeclareOp>() != nullptr) {
        const auto dwConvFilterContent = origOp.filter().getDefiningOp<Const::DeclareOp>().content();

        // Weights reverse according to ngraph implementation
        // https://github.com/openvinotoolkit/openvino/blob/745c8933bc67f0eaf7996848f5188521fdf50d14/ngraph/core/reference/include/ngraph/runtime/reference/convolution_backprop_data.hpp#L268
        // TODO: implement this reversing via lazy constant folding mechanism (JIRA:
        // https://jira.devtools.intel.com/browse/EISW-28339)
        SmallVector<float16> reversedVals(dwConvFilterContent.getValues<float16>());
        size_t spatialDims = filterShape[Dims4D::Filter::KX] * filterShape[Dims4D::Filter::KY];
        for (auto it = reversedVals.begin(); it < reversedVals.end(); it += spatialDims) {
            std::reverse(it, it + spatialDims);
        }

        SmallVector<int64_t> convFilterShape{filterShape[Dims4D::Filter::IC], filterShape[Dims4D::Filter::OC],
                                             filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]};

        const auto newConstType = dataStorageType.changeShape(ShapeRef(convFilterShape));
        const auto newDataStorageType =
                dataStorageType.changeElemType(mlir::Float16Type::get(getContext())).cast<mlir::RankedTensorType>();
        const auto dataAttr = mlir::DenseElementsAttr::get(newDataStorageType, makeArrayRef(reversedVals));
        const auto content = Const::ContentAttr::get(dataAttr).convertElemType(elemType).transpose(DimsOrder::IOYX);
        auto convFilter = rewriter.create<Const::DeclareOp>(origOp.getLoc(), newConstType, content);

        rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origOp, newUpsamplingOp.output(), convFilter.output(), nullptr,
                                                       strides, padsBegin, padsEnd, dilations, nullptr);
    } else {
        SmallVector<int64_t> revSeqFilterShape{filterShape[Dims4D::Filter::OC] * filterShape[Dims4D::Filter::IC],
                                               filterShape[Dims4D::Filter::KY] * filterShape[Dims4D::Filter::KX]};
        auto revSeqReshapedFilter = rewriter.create<IE::ReshapeOp>(origOp.getLoc(), origOp.filter(), nullptr, false,
                                                                   getIntArrayAttr(getContext(), revSeqFilterShape));

        SmallVector<int32_t> seqLengthVector;
        for (int batch = 0; batch < filterShape[Dims4D::Filter::OC] * filterShape[Dims4D::Filter::IC]; batch++)
            seqLengthVector.append(1, filterShape[Dims4D::Filter::KY] * filterShape[Dims4D::Filter::KX]);

        SmallVector<int64_t> seqLengthVectorShape{(int64_t)seqLengthVector.size()};
        auto dataStorageTypeRS =
                mlir::RankedTensorType::get(to_small_vector(seqLengthVectorShape),
                                            mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signed));
        const auto revSeqDataAttr = mlir::DenseElementsAttr::get(dataStorageTypeRS, makeArrayRef(seqLengthVector));
        const auto revSeqContent = Const::ContentAttr::get(revSeqDataAttr);
        auto seqLengthConst =
                rewriter.create<Const::DeclareOp>(origOp.getLoc(), revSeqContent.getType(), revSeqContent);

        auto reversedFilter = rewriter.create<IE::ReverseSequenceOp>(origOp.getLoc(), revSeqReshapedFilter.output(),
                                                                     seqLengthConst, 1, 0);

        SmallVector<int64_t> convFilterShape{filterShape[Dims4D::Filter::OC], filterShape[Dims4D::Filter::IC],
                                             filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]};

        auto reshapedFilter = rewriter.create<IE::ReshapeOp>(origOp.getLoc(), reversedFilter.output(), nullptr, false,
                                                             getIntArrayAttr(getContext(), convFilterShape));

        auto convFilter =
                rewriter.create<IE::TransposeOp>(origOp->getLoc(), reshapedFilter.output(), nullptr,
                                                 mlir::AffineMapAttr::get(DimsOrder::IOYX.toAffineMap(getContext())));

        rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origOp, newUpsamplingOp.output(), convFilter.output(), nullptr,
                                                       strides, padsBegin, padsEnd, dilations, nullptr);
    }

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
        const auto inputShape = getShape(conv.feature());
        return inputShape.size() != 4;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::DeconvolutionOp>(isLegalConvOp);
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::UpsamplingOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ReverseSequenceOp>();
    target.addLegalOp<IE::TransposeOp>();

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
