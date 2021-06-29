//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

class ExpandActivationChannels final : public IE::ExpandActivationChannelsBase<ExpandActivationChannels> {
public:
    explicit ExpandActivationChannels(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    class MaxPoolRewriter;
    class AvgPoolRewriter;
    class ConvolutionRewriter;

private:
    void safeRunOnFunc() final;
};

static Shape calculatePadShape(const mlir::ShapedType& origType) {
    const auto C = Dim(1);
    const auto origShape = getShape(origType);
    Shape padsEndArrayShape(checked_cast<size_t>(origType.getRank()), 0);
    const Bit typeSizeInBits = getElemTypeSize(origType);
    const int64_t CHANNELS_DIVIDER = 128 / typeSizeInBits.count();
    const auto extendedChannels = divUp(checked_cast<int64_t>(origShape[C]), CHANNELS_DIVIDER) * CHANNELS_DIVIDER;
    padsEndArrayShape[C] = extendedChannels - origShape[C];

    return padsEndArrayShape;
}

//
// GeneralRewriter
// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input

mlir::LogicalResult generalRewrite(mlir::Operation* layer, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, const int32_t)> opCreator) {
    mlir::MLIRContext* ctx = layer->getContext();

    const auto inputType = layer->getOperand(0).getType().cast<mlir::ShapedType>();
    Shape inPadsEndArrayShape = calculatePadShape(inputType);

    const auto outputType = layer->getResult(0).getType().cast<mlir::ShapedType>();
    Shape outPadsEndArrayShape = calculatePadShape(outputType);

    static const auto C = Dim(1);
    if (inPadsEndArrayShape[C] == 0 && outPadsEndArrayShape[C] == 0) {
        // It's ok, shapes of input and output tensors already satisfied hardware requirements
        // there is no need to extend channels count
        return matchFailed(rewriter, layer, "Channels have already been aligned. Nothing to do.");
    }
    auto padsBegin = getInt32ArrayAttr(ctx, mlir::SmallVector<uint32_t>(checked_cast<size_t>(inputType.getRank()), 0));
    auto padsEnd = getInt32ArrayAttr(ctx, inPadsEndArrayShape);

    mlir::Value paddedInput;
    if (inPadsEndArrayShape[C] == 0) {
        // use original input directly, padding is not required
        paddedInput = layer->getOperand(0);
    } else {
        paddedInput = rewriter.create<IE::ExpandOp>(layer->getLoc(), layer->getOperand(0), padsBegin, padsEnd);
    }
    auto newOp = opCreator(paddedInput, checked_cast<int32_t>(outPadsEndArrayShape[C]));
    if (newOp == nullptr) {
        return matchFailed(rewriter, layer, "Failed to create replacing operation");
    }

    const auto inputShape = getShape(inputType);
    SmallVector<int64_t> offsets(inputShape.size(), 0);
    SmallVector<int64_t> strides(inputShape.size(), 1);
    auto shape = outputType.getShape();

    auto subTensorOp = rewriter.create<mlir::SubTensorOp>(
            layer->getLoc(), layer->getResult(0).getType(), newOp->getResult(0), mlir::ValueRange{}, mlir::ValueRange{},
            mlir::ValueRange{}, getInt64ArrayAttr(ctx, offsets), getInt64ArrayAttr(ctx, shape),
            getInt64ArrayAttr(ctx, strides));

    rewriter.replaceOp(layer, {subTensorOp});
    return mlir::success();
}

//
// MaxPoolProcessor
//

class ExpandActivationChannels::MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
    }
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandActivationChannels::MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    auto func = [&rewriter, &origOp](mlir::Value expandedInput, const int32_t) -> IE::MaxPoolOp {
        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), expandedInput, origOp.kernel_size(), origOp.strides(),
                                              origOp.pads_begin(), origOp.pads_end(), origOp.rounding_type());
    };
    return generalRewrite(origOp, rewriter, func);
}

//
// AvgPoolProcessor
//

class ExpandActivationChannels::AvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandActivationChannels::AvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    auto func = [&rewriter, &origOp](mlir::Value expandedInput, const int32_t) -> IE::AvgPoolOp {
        return rewriter.create<IE::AvgPoolOp>(origOp.getLoc(), expandedInput, origOp.kernel_size(), origOp.strides(),
                                              origOp.pads_begin(), origOp.pads_end(), origOp.rounding_type());
    };
    return generalRewrite(origOp, rewriter, func);
}

//
// ConvolutionProcessor
//

class ExpandActivationChannels::ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

static std::vector<float> populateExpandedData(const ShapeRef& srcShape, const ShapeRef& dstShape,
                                               const vpux::details::ConstContentRange<float>& origWeightsData) {
    const auto dimOC = Dim(0);
    const auto dimIC = Dim(1);
    const auto dimKY = Dim(2);
    const auto dimKX = Dim(3);
    const auto srcOC = srcShape[dimOC];
    const auto srcIC = srcShape[dimIC];
    const auto srcKY = srcShape[dimKY];
    const auto srcKX = srcShape[dimKX];

    const auto dstOC = dstShape[dimOC];
    const auto dstIC = dstShape[dimIC];
    const auto dstKY = dstShape[dimKY];
    const auto dstKX = dstShape[dimKX];

    const auto srcOCStride = srcKY * srcKX * srcIC;
    const auto srcICStride = srcKY * srcKX;
    const auto srcKYStride = srcKX;
    const auto srcKXStride = 1;

    const auto dstOCStride = dstKY * dstKX * dstIC;
    const auto dstICStride = dstKY * dstKX;
    const auto dstKYStride = dstKX;
    const auto dstKXStride = 1;

    // 1. Create vector with enough data for expanded tensor.
    // 2. Populate that vector with zeros.
    // 3. Insert original weights according to strides.
    // Please note that there's no re-pack here. Both source and destination tensors have OIYX layouts.
    std::vector<float> expandedWeightsData(dstOC * dstIC * dstKY * dstKX, 0);
    for (long oc = 0; oc < srcOC; oc++) {
        for (long ic = 0; ic < srcIC; ic++) {
            for (long ky = 0; ky < srcKY; ky++) {
                for (long kx = 0; kx < srcKX; kx++) {
                    size_t srcIdx = oc * srcOCStride + ic * srcICStride + ky * srcKYStride + kx * srcKXStride;
                    size_t dstIdx = oc * dstOCStride + ic * dstICStride + ky * dstKYStride + kx * dstKXStride;
                    expandedWeightsData[dstIdx] = origWeightsData[srcIdx];
                }
            }
        }
    }

    return expandedWeightsData;
}

static mlir::Value expandConstant(mlir::Value origTensor, const unsigned int expandSize, const Dim& axis,
                                  mlir::Location loc, mlir::PatternRewriter& rewriter) {
    if (expandSize <= 0) {
        // nothing to be done
        return origTensor;
    }

    const auto origShape = getShape(origTensor);
    auto targetShape = origShape.toValues();
    targetShape[axis] += expandSize;

    auto weightsConst = origTensor.getDefiningOp<ConstantInterface>();
    if (weightsConst == nullptr) {
        VPUX_THROW("Weights do not provide constant interface");
    }

    const auto origWeightsData = weightsConst.getContent().getValues<float>();
    const auto expandedWeightsData = populateExpandedData(origShape, targetShape, origWeightsData);

    // create tensor from expanded data
    const auto originalFilterType = origTensor.getType().cast<mlir::ShapedType>();
    const auto expandedFilterStorageType = mlir::RankedTensorType::get(targetShape.raw(), rewriter.getF32Type());
    const auto expandedFilterAttr =
            mlir::DenseElementsAttr::get(expandedFilterStorageType, makeArrayRef(expandedWeightsData));
    const auto expandedFilterType = mlir::RankedTensorType::get(targetShape.raw(), originalFilterType.getElementType());
    auto paddedConstant = rewriter.create<IE::ConstantOp>(loc, expandedFilterType, expandedFilterAttr);
    return paddedConstant.output();
}

mlir::LogicalResult ExpandActivationChannels::ConvolutionRewriter::matchAndRewrite(
        IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    auto func = [&rewriter, &origOp](mlir::Value expandedInput, const int32_t expandOutChanSize) -> IE::ConvolutionOp {
        // we have to expand channels count for kernel tensor as well
        static const auto OutChan = Dim(0);
        static const auto InChan = Dim(1);

        const auto filterType = origOp.filter().getType().cast<mlir::ShapedType>();
        const auto filterShape = getShape(filterType);
        const auto newInputType = expandedInput.getType().cast<mlir::ShapedType>();
        const auto newInputShape = getShape(newInputType);

        // extend input channels
        const auto expandInChanSize = checked_cast<unsigned int>(newInputShape[InChan] - filterShape[InChan]);
        mlir::Value paddedInFilter =
                expandConstant(origOp.filter(), expandInChanSize, InChan, origOp.getLoc(), rewriter);

        // extend output channels
        mlir::Value paddedFilter =
                expandConstant(paddedInFilter, expandOutChanSize, OutChan, origOp.getLoc(), rewriter);

        // extend biases
        mlir::Value paddedBiases;
        if (origOp.bias() != nullptr) {
            if (expandOutChanSize > 0) {
                const auto biasType = origOp.bias().getType().cast<mlir::ShapedType>();
                const auto biasShape = getShape(biasType);

                auto additionalZerosShape = to_small_vector(biasShape);
                additionalZerosShape[checked_cast<size_t>(InChan.ind())] +=
                        checked_cast<unsigned int>(expandOutChanSize);

                const auto additionalZerosTensorType =
                        mlir::RankedTensorType::get(additionalZerosShape, biasType.getElementType());

                // FIXME concatenation must be used in order to avoid bias data duplication
                std::vector<float> extendedBiasData(additionalZerosTensorType.getNumElements(), 0);
                auto biasConst = origOp.bias().getDefiningOp<ConstantInterface>();
                if (biasConst == nullptr) {
                    VPUX_THROW("Bias does not provide constant interface");
                }

                for (auto p : enumerate(biasConst.getContent().getValues<float>())) {
                    extendedBiasData.at(p.index()) = p.value();
                }

                const auto zerosStorageType = mlir::RankedTensorType::get(additionalZerosShape, rewriter.getF32Type());
                const auto extendedBiasAttr =
                        mlir::DenseElementsAttr::get(zerosStorageType, makeArrayRef(extendedBiasData));

                paddedBiases =
                        rewriter.create<IE::ConstantOp>(origOp.getLoc(), additionalZerosTensorType, extendedBiasAttr);
            } else {
                paddedBiases = origOp.bias();
            }
        }
        auto res = rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), expandedInput, paddedFilter, paddedBiases,
                                                      origOp.strides(), origOp.pads_begin(), origOp.pads_end(),
                                                      origOp.dilations());
        return res;
    };
    return generalRewrite(origOp, rewriter, func);
}

//
// safeRunOnFunc
//

void ExpandActivationChannels::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto channelsSatisfyDPUreq = [&](mlir::Operation* op) {
        if (mlir::isa<IE::MaxPoolOp, IE::AvgPoolOp, IE::ConvolutionOp>(op)) {
            static const auto C = Dim(1);
            const auto firstInput = op->getOperand(0);
            const auto inType = firstInput.getType().cast<mlir::ShapedType>();
            const auto inShape = getShape(firstInput);

            const Bit inTypeSizeInBits = getElemTypeSize(inType);
            const int64_t IN_CHANNELS_DIVIDER = 128 / inTypeSizeInBits.count();
            if (inType.getRank() != 4 || inShape[C] % IN_CHANNELS_DIVIDER != 0) {
                return false;
            }

            const auto firstOutput = op->getResult(0);
            const auto outType = firstOutput.getType().cast<mlir::ShapedType>();
            const auto outShape = getShape(firstOutput);

            const Bit outTypeSizeInBits = getElemTypeSize(outType);
            const int64_t OUT_CHANNELS_DIVIDER = 128 / outTypeSizeInBits.count();
            if (outType.getRank() != 4 || outShape[C] % OUT_CHANNELS_DIVIDER != 0) {
                return false;
            }
        }
        return true;
    };
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(channelsSatisfyDPUreq);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(channelsSatisfyDPUreq);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(channelsSatisfyDPUreq);
    target.addLegalOp<mlir::SubTensorOp>();
    target.addLegalOp<IE::ConstantOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::ExpandOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MaxPoolRewriter>(&ctx, _log);
    patterns.insert<AvgPoolRewriter>(&ctx, _log);
    patterns.insert<ConvolutionRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createExpandActivationChannelsPass(Logger log) {
    return std::make_unique<ExpandActivationChannels>(log);
}
