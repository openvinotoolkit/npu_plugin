//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ReduceSumToConvRewriter
//

class ReduceSumToConvRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    ReduceSumToConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("ReduceSumToConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isValidShape(vpux::ShapeRef inputShape, int64_t alignment, Logger log) const;
    bool isSupportedReduceSum(IE::ReduceSumOp origOp, int64_t alignment, Logger log) const;

    IE::ConvolutionOp createConvolution(mlir::Value activation, mlir::Value weights, mlir::Location newLoc,
                                        mlir::PatternRewriter& rewriter) const;

    // Functions for unaligned input
    void convertReduceSumToConvolutionForUnalignedInput(IE::ReduceSumOp origOp, int64_t factor,
                                                        mlir::PatternRewriter& rewriter) const;
    IE::AffineReshapeOp buildReshape(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                                     mlir::PatternRewriter& rewriter) const;
    IE::AffineReshapeOp reshapeInput(mlir::Value input, const int64_t factor, mlir::PatternRewriter& rewriter) const;
    IE::AffineReshapeOp reshapeOutput(IE::ReduceSumOp origOp, mlir::Value convOutput,
                                      mlir::PatternRewriter& rewriter) const;
    mlir::Value createConvFilterForReshapedActivation(mlir::Value activation, int64_t factor,
                                                      mlir::PatternRewriter& rewriter) const;

    // Functions for channel aligned input
    void convertReduceSumToConvolutionForChannelAlignedInput(IE::ReduceSumOp origOp,
                                                             mlir::PatternRewriter& rewriter) const;
    mlir::Value createConvFilterForChannelAlignedActivation(mlir::Value activation,
                                                            mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

bool ReduceSumToConvRewriter::isValidShape(vpux::ShapeRef inputShape, int64_t alignment, Logger log) const {
    if (inputShape.size() != 4) {
        log.trace("Only support 4D ReduceSum");
        return false;
    }

    if (inputShape[Dims4D::Act::N] != 1) {
        log.trace("Batch must be equal to 1");
        return false;
    }

    auto inChannel = inputShape[Dims4D::Act::C];
    if (inChannel >= alignment && inChannel % alignment == 0) {
        // Input channel is aligned
        return true;
    }

    if (inChannel < alignment && alignment % inChannel == 0) {
        // Input channel is not aligned
        // This can by handled by re-arranging [factor] lines on height to channel and avoid expand
        // This requires [alignment] can be divided by [IC] and height can be divived by [factor]
        // [IH] is divided by C to facilitate the construction of the convolution weights
        auto factor = alignment / inChannel;
        return inputShape[Dims4D::Act::H] % factor == 0;
    }

    return false;
}

bool ReduceSumToConvRewriter::isSupportedReduceSum(IE::ReduceSumOp origOp, int64_t alignment, Logger log) const {
    // Check shape
    const auto inputShape = getShape(origOp.input());
    if (!isValidShape(inputShape, alignment, log)) {
        log.trace("Shape is invalid {0} at {1}", origOp->getName(), origOp->getLoc());
        return false;
    }

    // Check reduce axis
    auto axes = parseIntArrayAttr<int64_t>(origOp.axes_value().value());
    if (axes.size() != 1) {
        log.trace("Only support ReduceSum reduce on one dimension");
        return false;
    }

    auto reduceAxis = axes[0];
    if (reduceAxis != Dims4D::Act::C.ind()) {
        log.trace("Only support ReduceSum reduce on channel");
        return false;
    }

    // Check keep_dims
    if (!origOp.keep_dims()) {
        log.trace("Only support ReduceSum when keep_dims is true");
        return false;
    }

    return true;
}

IE::ConvolutionOp ReduceSumToConvRewriter::createConvolution(mlir::Value activation, mlir::Value weights,
                                                             mlir::Location newLoc,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    return rewriter.create<IE::ConvolutionOp>(newLoc, activation, weights, nullptr, strides, kernelPadsBegin,
                                              kernelPadsEnd, dilations, nullptr);
}

IE::AffineReshapeOp ReduceSumToConvRewriter::buildReshape(mlir::Location loc, mlir::Value input,
                                                          ArrayRef<int64_t> targetShape,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto srcType = input.getType().cast<vpux::NDTypeInterface>();
    const auto dstType = srcType.changeShape(ShapeRef(targetShape));
    SmallVector<SmallVector<int64_t>> reassociationMap(targetShape.size());
    for (const auto& dimIdx : irange(reassociationMap.size())) {
        reassociationMap[dimIdx].push_back(dimIdx);
    }
    const auto reassociationMapAttr = getIntArrayOfArray(ctx, reassociationMap);
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape);
    auto reshapeOp = rewriter.create<IE::AffineReshapeOp>(loc, dstType, input, reassociationMapAttr, targetShapeAttr);

    return reshapeOp;
}

IE::AffineReshapeOp ReduceSumToConvRewriter::reshapeInput(mlir::Value input, const int64_t factor,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto origInShape = getShape(input);
    const auto batch = origInShape[Dims4D::Act::N];
    const auto channels = origInShape[Dims4D::Act::C] * factor;  // New channel should be 16
    const auto height = origInShape[Dims4D::Act::H] / factor;
    const auto width = origInShape[Dims4D::Act::W];

    const SmallVector<int64_t> targetShape = {batch, channels, height, width};
    const auto reshapedLoc = appendLoc(input.getLoc(), "_input_reshape");
    return buildReshape(reshapedLoc, input, makeArrayRef(targetShape), rewriter);
}

IE::AffineReshapeOp ReduceSumToConvRewriter::reshapeOutput(IE::ReduceSumOp origOp, mlir::Value convOutput,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto origOutShape = getShape(origOp.output());
    const SmallVector<int64_t> targetShape = to_small_vector(origOutShape.raw());

    const auto reshapedLoc = appendLoc(origOp.getLoc(), "_output_reshape");
    return buildReshape(reshapedLoc, convOutput, makeArrayRef(targetShape), rewriter);
}

//
// For example, a ReduceSum operation with 1x4x8x8@NCHW input tensor
// Step 1. Align input channel by re-arrange input shape to 1x16x2x8@NCHW
// Step 2. Create the OCxICx1x1 (OC = 4, IC = 16) convolution filter, the IC x OC filter should be:
// Filter #0
// 1 0 0 0 | 1 0 0 0 | 1 0 0 0 | 1 0 0 0
// Filter #1
// 0 1 0 0 | 0 1 0 0 | 0 1 0 0 | 0 1 0 0
// Filter #2
// 0 0 1 0 | 0 0 1 0 | 0 0 1 0 | 0 0 1 0
// Filter #3
// 0 0 0 1 | 0 0 0 1 | 0 0 0 1 | 0 0 0 1
// Step 3. Create convoulution with actvation and weights created in step 1 & 2, result is 1x4x2x8@NCHW
// Step 4. Re-arrange convoulution 1x4x2x8@NCHW result to original ReduceSum output shape 1x1x8x8@NCHW
//

mlir::Value ReduceSumToConvRewriter::createConvFilterForReshapedActivation(mlir::Value activation, int64_t factor,
                                                                           mlir::PatternRewriter& rewriter) const {
    const auto IC = getShape(activation)[Dims4D::Act::C];
    const auto KX = 1;
    const auto KY = 1;
    const auto OC = factor;

    const Shape weightShape = {OC, IC, KX, KY};

    SmallVector<float> weights(weightShape.totalSize(), .0f);

    // assign values
    for (auto j = 0; j < OC; ++j) {
        for (auto i = 0; i < IC / factor; ++i) {
            const auto index = j * IC + j + i * factor;
            weights[index] = 1.0f;
        }
    }

    const DimsOrder weighOrder = DimsOrder::OIYX;

    return VPU::buildWeightsConst(ShapeRef(weightShape), weighOrder, makeArrayRef(weights), activation, rewriter);
}

void ReduceSumToConvRewriter::convertReduceSumToConvolutionForUnalignedInput(IE::ReduceSumOp origOp, int64_t factor,
                                                                             mlir::PatternRewriter& rewriter) const {
    // Reshape activation
    auto reshapedActivation = reshapeInput(origOp.input(), factor, rewriter);

    // Create convolution filiter
    auto weights = createConvFilterForReshapedActivation(reshapedActivation.output(), factor, rewriter);

    // Create convolution
    const auto convLoc = appendLoc(origOp->getLoc(), "_convolution");
    auto conv = createConvolution(reshapedActivation, weights, convLoc, rewriter);

    // Reshape output
    auto result = reshapeOutput(origOp, conv.output(), rewriter);

    rewriter.replaceOp(origOp, result.output());
}

//
// For example, a ReduceSum operation with 1x16x8x8@NCHW input tensor
// Create 1x16x1x1 convolution filter, the weights value should be:
// 1 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1
//
mlir::Value ReduceSumToConvRewriter::createConvFilterForChannelAlignedActivation(
        mlir::Value activation, mlir::PatternRewriter& rewriter) const {
    const auto IC = getShape(activation)[Dims4D::Act::C];
    const auto KX = 1;
    const auto KY = 1;
    const auto OC = 1;

    const Shape weightShape = {OC, IC, KX, KY};

    SmallVector<float> weights(weightShape.totalSize(), .0f);

    // assign values
    for (auto i = 0; i < IC; ++i) {
        weights[i] = 1.0f;
    }

    const DimsOrder weighOrder = DimsOrder::OIYX;

    return VPU::buildWeightsConst(ShapeRef(weightShape), weighOrder, makeArrayRef(weights), activation, rewriter);
}

void ReduceSumToConvRewriter::convertReduceSumToConvolutionForChannelAlignedInput(
        IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const {
    // Create convolution filiter
    auto weights = createConvFilterForChannelAlignedActivation(origOp.input(), rewriter);

    // Create convolution
    const auto convLoc = appendLoc(origOp->getLoc(), "_convolution");
    auto conv = createConvolution(origOp.input(), weights, convLoc, rewriter);

    rewriter.replaceOp(origOp, conv.output());
}

mlir::LogicalResult ReduceSumToConvRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto inType = origOp.input().getType().dyn_cast<vpux::NDTypeInterface>();
    const auto inputElemType = inType.getElementType();
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inputElemType);
    const ShapeRef origInShape = inType.getShape();

    if (!isSupportedReduceSum(origOp, inAlignment, _log)) {
        return mlir::failure();
    }

    const auto origLoc = origOp->getLoc();
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origLoc);

    auto inChannel = origInShape[Dims4D::Act::C];
    auto isInputAligned = (inChannel % inAlignment == 0);
    if (isInputAligned) {
        convertReduceSumToConvolutionForChannelAlignedInput(origOp, rewriter);
    } else {
        // In below, it is going to convert channel unaligned input ReduceSum to Convolution
        VPUX_THROW_WHEN(inAlignment % inChannel != 0, "Alignment {0} can not be divided by IC {1}", inAlignment,
                        inChannel);
        // Will re-arranging [factor] lines on height to channel to avoid expand
        auto factor = inAlignment / inChannel;
        VPUX_THROW_WHEN(origInShape[Dims4D::Act::H] % factor != 0, "Height {0} can not be divided by factor {1}",
                        origInShape[Dims4D::Act::H], factor);

        convertReduceSumToConvolutionForUnalignedInput(origOp, factor, rewriter);
    }
    _log.trace("[{0}] Successfully convert ReduceSum to Convolution '{1}'", getDebugName(), origLoc);
    return mlir::success();
}

//
// ConvertReduceSumToConvPass
//

class ConvertReduceSumToConvPass final : public IE::ConvertReduceSumToConvBase<ConvertReduceSumToConvPass> {
public:
    explicit ConvertReduceSumToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertReduceSumToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    // Convert ReduceSum to Convolution operation is optimum solution in case reduce axis is C
    mlir::RewritePatternSet pattern(&ctx);
    pattern.add<ReduceSumToConvRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(pattern), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReduceSumToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReduceSumToConvPass(Logger log) {
    return std::make_unique<ConvertReduceSumToConvPass>(log);
}
