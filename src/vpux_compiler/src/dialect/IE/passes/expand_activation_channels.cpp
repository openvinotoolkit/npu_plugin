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
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

Shape calcPadsEnd(mlir::ShapedType origType) {
    static const auto C = Dim(1);

    const auto origShape = getShape(origType);

    const Bit typeSizeInBits = getElemTypeSize(origType);
    const int64_t CHANNELS_DIVIDER = 128 / typeSizeInBits.count();

    const auto extendedChannels = divUp(origShape[C], CHANNELS_DIVIDER) * CHANNELS_DIVIDER;

    Shape padsEndArrayShape(origShape.size(), 0);
    padsEndArrayShape[C] = extendedChannels - origShape[C];

    return padsEndArrayShape;
}

//
// GeneralRewriter
// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input
//

mlir::LogicalResult generalRewrite(mlir::Operation* layer, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator) {
    auto* ctx = layer->getContext();

    const auto inputType = layer->getOperand(0).getType().cast<mlir::ShapedType>();
    const auto inPadsEnd = calcPadsEnd(inputType);

    const auto outputType = layer->getResult(0).getType().cast<mlir::ShapedType>();
    const auto outPadsEnd = calcPadsEnd(outputType);

    static const auto C = Dim(1);

    if (inPadsEnd[C] == 0 && outPadsEnd[C] == 0) {
        // It's ok, shapes of input and output tensors already satisfied hardware requirements
        // there is no need to extend channels count
        return matchFailed(rewriter, layer, "Channels have already been aligned. Nothing to do.");
    }

    mlir::Value paddedInput;

    if (inPadsEnd[C] == 0) {
        // use original input directly, padding is not required
        paddedInput = layer->getOperand(0);
    } else {
        const SmallVector<uint32_t> inPadsBegin(inPadsEnd.size(), 0);

        paddedInput =
                rewriter.create<IE::ExpandOp>(layer->getLoc(), layer->getOperand(0),
                                              getInt32ArrayAttr(ctx, inPadsBegin), getInt32ArrayAttr(ctx, inPadsEnd));
    }

    auto newOp = opCreator(paddedInput, outPadsEnd[C]);

    const auto outShape = outputType.getShape();
    const SmallVector<int64_t> offsets(outShape.size(), 0);
    const SmallVector<int64_t> strides(outShape.size(), 1);

    rewriter.replaceOpWithNewOp<mlir::SubTensorOp>(layer, layer->getResult(0).getType(), newOp->getResult(0),
                                                   mlir::ValueRange{}, mlir::ValueRange{}, mlir::ValueRange{},
                                                   getInt64ArrayAttr(ctx, offsets), getInt64ArrayAttr(ctx, outShape),
                                                   getInt64ArrayAttr(ctx, strides));

    return mlir::success();
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto opCreator = [&](mlir::Value expandedInput, int64_t) -> mlir::Operation* {
        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), expandedInput, origOp.kernel_size(), origOp.strides(),
                                              origOp.pads_begin(), origOp.pads_end(), origOp.rounding_type());
    };

    return generalRewrite(origOp, rewriter, opCreator);
}

//
// AvgPoolRewriter
//

class AvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto opCreator = [&](mlir::Value expandedInput, int64_t) -> mlir::Operation* {
        return rewriter.create<IE::AvgPoolOp>(origOp.getLoc(), expandedInput, origOp.kernel_size(), origOp.strides(),
                                              origOp.pads_begin(), origOp.pads_end(), origOp.rounding_type());
    };

    return generalRewrite(origOp, rewriter, opCreator);
}

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto filter_out_channel_dim = IE::ConvolutionOp::filter_out_channel_dim();
        const auto filter_in_channel_dim = IE::ConvolutionOp::filter_in_channel_dim();
        const auto act_channel_dim = IE::ConvolutionOp::act_channel_dim();

        const auto newInputShape = getShape(expandedInput);

        const auto filterShape = getShape(origOp.filter());

        const auto inChanPadEnd = newInputShape[act_channel_dim] - filterShape[filter_in_channel_dim];

        mlir::Value paddedFilter;

        if (inChanPadEnd == 0 && outChanPadEnd == 0) {
            paddedFilter = origOp.filter();
        } else {
            const SmallVector<uint32_t> filterPadsBegin(filterShape.size(), 0);

            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[filter_out_channel_dim] = checked_cast<uint32_t>(outChanPadEnd);
            filterPadsEnd[filter_in_channel_dim] = checked_cast<uint32_t>(inChanPadEnd);

            const auto padValue = getFP32Attr(getContext(), 0.0f);

            paddedFilter = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.filter(), nullptr, nullptr,
                                                            nullptr, getInt32ArrayAttr(getContext(), filterPadsBegin),
                                                            getInt32ArrayAttr(getContext(), filterPadsEnd.raw()),
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
                biasPadsEnd[act_channel_dim] = checked_cast<uint32_t>(outChanPadEnd);

                const auto padValue = getFP32Attr(getContext(), 0.0f);

                paddedBiases = rewriter.createOrFold<IE::PadOp>(origOp->getLoc(), origOp.bias(), nullptr, nullptr,
                                                                nullptr, getInt32ArrayAttr(getContext(), biasPadsBegin),
                                                                getInt32ArrayAttr(getContext(), biasPadsEnd.raw()),
                                                                padValue, IE::PadMode::CONSTANT);
            }
        }

        return rewriter.create<IE::ConvolutionOp>(origOp.getLoc(), expandedInput, paddedFilter, paddedBiases,
                                                  origOp.strides(), origOp.pads_begin(), origOp.pads_end(),
                                                  origOp.dilations());
    };

    return generalRewrite(origOp, rewriter, opCreator);
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

    const auto channelsSatisfyDPUreq = [](mlir::Operation* op) {
        const auto act_channel_dim = IE::ConvolutionOp::act_channel_dim();

        const auto firstInput = op->getOperand(0);
        const auto inType = firstInput.getType().cast<mlir::ShapedType>();
        const auto inShape = getShape(firstInput);

        const Bit inTypeSizeInBits = getElemTypeSize(inType);
        const int64_t IN_CHANNELS_DIVIDER = 128 / inTypeSizeInBits.count();
        if (inType.getRank() != 4 || inShape[act_channel_dim] % IN_CHANNELS_DIVIDER != 0) {
            return false;
        }

        const auto firstOutput = op->getResult(0);
        const auto outType = firstOutput.getType().cast<mlir::ShapedType>();
        const auto outShape = getShape(firstOutput);

        const Bit outTypeSizeInBits = getElemTypeSize(outType);
        const int64_t OUT_CHANNELS_DIVIDER = 128 / outTypeSizeInBits.count();
        if (outType.getRank() != 4 || outShape[act_channel_dim] % OUT_CHANNELS_DIVIDER != 0) {
            return false;
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(channelsSatisfyDPUreq);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(channelsSatisfyDPUreq);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(channelsSatisfyDPUreq);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::PadOp>();
    target.addLegalOp<mlir::SubTensorOp>();

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
    return std::make_unique<ExpandActivationChannelsPass>(log);
}
