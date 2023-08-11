//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

class DPUExpandRewriter final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    DPUExpandRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isEligibleForConversion(IE::ExpandOp expandOp) const;
    IE::AffineReshapeOp buildReshape(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                                     mlir::PatternRewriter& rewriter) const;
    IE::AffineReshapeOp reshapeInput(IE::ExpandOp origOp, const int64_t channelAlignment,
                                     mlir::PatternRewriter& rewriter) const;
    IE::AffineReshapeOp reshapeOutput(IE::ExpandOp origOp, mlir::Value convOutput,
                                      mlir::PatternRewriter& rewriter) const;
    mlir::Value composeWeights(IE::ExpandOp origOp, const int64_t convolutionAlignment,
                               mlir::PatternRewriter& rewriter) const;
    IE::ConvolutionOp buildConvolution(IE::ExpandOp expandOp, mlir::Value activation, mlir::Value weights,
                                       mlir::PatternRewriter& rewriter) const;
    Logger _log;
};

int64_t greatestCommonDivisor(int64_t term1, int64_t term2) {
    while (term1 != term2) {
        if (term1 > term2) {
            term1 = term1 - term2;
        } else {
            term2 = term2 - term1;
        }
    }
    return term1;
}

int64_t leastCommonMultiple(int64_t term1, int64_t term2) {
    VPUX_THROW_WHEN(term1 <= 0 || term2 <= 0,
                    "This implementation of LCM expects two positive integers, got {0} and {1}", term1, term2);
    return term1 * term2 / greatestCommonDivisor(term1, term2);
}

int64_t calculateAlignment(const vpux::NDTypeInterface expandInType) {
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(expandInType.getElementType());
    const auto expandInChannels = expandInType.getShape()[Dims4D::Act::C];
    const auto leastChannelMultiple = leastCommonMultiple(channelAlignment, expandInChannels);
    return leastChannelMultiple / expandInChannels;
}

Shape calculateWeightsShape(ShapeRef expandInShape, ShapeRef expandOutShape, const int64_t alignment) {
    const int64_t kernelOutputChannels = expandOutShape[Dims4D::Act::C] * alignment;
    const int64_t kernelInputChannels = expandInShape[Dims4D::Act::C] * alignment;
    const int64_t kernelY = 1;
    const int64_t kernelX = 1;
    return Shape{kernelOutputChannels, kernelInputChannels, kernelY, kernelX};
}

bool DPUExpandRewriter::isEligibleForConversion(IE::ExpandOp expandOp) const {
    const auto expandInType = expandOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto expandOutType = expandOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto supportedLayout = DimsOrder::NHWC;
    const auto expandInLayout = expandInType.getDimsOrder();
    if (expandInLayout != supportedLayout) {
        _log.trace("[{0}]: Expand at {1} has {2} input layout, expected {3}", getDebugName(), expandOp.getLoc(),
                   expandInLayout, supportedLayout);
        return false;
    }
    const auto expandOutLayout = expandOutType.getDimsOrder();
    if (expandOutLayout != supportedLayout) {
        _log.trace("[{0}]: Expand at {1} has {2} output layout, expected {3}", getDebugName(), expandOp.getLoc(),
                   expandOutLayout, supportedLayout);
        return false;
    }
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.pads_beginAttr());
    if (expandPadsBegin.size() != 4) {
        _log.trace("[{0}]: Expand at {1} has {2}-d start padding. Only 4-d shapes are supported", getDebugName(),
                   expandOp.getLoc(), expandPadsBegin.size());
        return false;
    }
    const auto isConflictingPadBegin = [](const int64_t pad) -> bool {
        return pad != 0;
    };
    if (std::any_of(expandPadsBegin.begin(), expandPadsBegin.end(), isConflictingPadBegin)) {
        _log.trace("[{0}]: Expand at {1} has {2} start padding. Expected to have [0, 0, 0, 0]", getDebugName(),
                   expandOp.getLoc(), expandPadsBegin);
        return false;
    }
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.pads_endAttr());
    if (expandPadsEnd.size() != 4) {
        _log.trace("[{0}]: Expand at {1} has {2}-d end padding. Only 4-d shapes are supported", getDebugName(),
                   expandOp.getLoc(), expandPadsEnd.size());
        return false;
    }
    if (expandPadsEnd[Dims4D::Act::N.ind()] != 0 || expandPadsEnd[Dims4D::Act::C.ind()] <= 0 ||
        expandPadsEnd[Dims4D::Act::H.ind()] != 0 || expandPadsEnd[Dims4D::Act::W.ind()] != 0) {
        _log.trace("[{0}]: Expand at {1} has {2} end padding. Expected to have [0, C, 0, 0]", getDebugName(),
                   expandOp.getLoc(), expandPadsEnd);
        return false;
    }
    const auto expandInShape = expandInType.getShape();
    if (expandInShape.size() != 4) {
        _log.trace("[{0}]: Expand at {1} has {2}-d shape. Only 4-d shapes are supported", getDebugName(),
                   expandOp.getLoc(), expandInShape.size());
        return false;
    }
    if (expandInShape[Dims4D::Act::N] != 1) {
        _log.trace("[{0}]: Expand at {1} has batch {2}. Expected to have 1", getDebugName(), expandOp.getLoc(),
                   expandInShape[Dims4D::Act::N]);
        return false;
    }
    const auto convolutionAlignment = calculateAlignment(expandInType);
    if (expandInShape[Dims4D::Act::W] % convolutionAlignment != 0) {
        _log.trace("[{0}]: Expand at {1} has width {2}. Width is expected to be a multiple of {3}", getDebugName(),
                   expandOp.getLoc(), expandInShape[Dims4D::Act::W], convolutionAlignment);
        return false;
    }
    if (!expandInType.getElementType().isF16()) {
        _log.trace("[{0}]: Expand at {1} has {2} element type. Only float16 is supported", getDebugName(),
                   expandOp.getLoc(), expandInType.getElementType());
        return false;
    }

    // Large tensors break scheduling.
    // See E#81001 for details.
    const auto expandOutShape = expandOutType.getShape();
    const auto weightShape = calculateWeightsShape(expandInShape, expandOutShape, convolutionAlignment);
    const auto elementSize = Byte(expandInType.getElemTypeSize()).count();
    const auto weightsSize =
            std::accumulate(weightShape.begin(), weightShape.end(), elementSize, std::multiplies<int64_t>());
    const Byte totalInputSize = expandInType.getTotalAllocSize();
    const Byte totalOutputSize = expandOutType.getTotalAllocSize();
    const Byte totalWeightsSize = Byte(weightsSize);
    const auto arch = VPU::getArch(expandOp.getOperation());
    const auto totalCMXSize = VPU::getTotalCMXSize(expandOp.getOperation());
    SmallVector<Byte> buffers = {totalInputSize, totalWeightsSize, totalOutputSize};
    const auto memReq = VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers);
    if (memReq > totalCMXSize) {
        _log.trace("[{0}]: Expand at {1} requires {2} bytes of CMX while only {3} bytes are available", getDebugName(),
                   expandOp.getLoc(), memReq, totalCMXSize);
        return false;
    }

    return true;
}

IE::AffineReshapeOp DPUExpandRewriter::buildReshape(mlir::Location loc, mlir::Value input,
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

IE::AffineReshapeOp DPUExpandRewriter::reshapeInput(IE::ExpandOp origOp, const int64_t channelAlignment,
                                                    mlir::PatternRewriter& rewriter) const {
    const auto origShape = getShape(origOp.input());
    const auto batch = origShape[Dims4D::Act::N];
    const auto channels = origShape[Dims4D::Act::C] * channelAlignment;
    const auto height = origShape[Dims4D::Act::H];
    const auto width = origShape[Dims4D::Act::W] / channelAlignment;

    const SmallVector<int64_t> targetShape = {batch, channels, height, width};
    const auto reshapedLoc = appendLoc(origOp.getLoc(), "reshape input for DPU expand");
    return buildReshape(reshapedLoc, origOp.input(), makeArrayRef(targetShape), rewriter);
}

IE::AffineReshapeOp DPUExpandRewriter::reshapeOutput(IE::ExpandOp origOp, mlir::Value convOutput,
                                                     mlir::PatternRewriter& rewriter) const {
    const Shape origShape = getShape(origOp.output()).toValues();
    const SmallVector<int64_t> targetShape = origShape.raw();

    const auto reshapedLoc = appendLoc(origOp.getLoc(), "reshape output for DPU expand");
    return buildReshape(reshapedLoc, convOutput, makeArrayRef(targetShape), rewriter);
}

// The idea is to create the following structure:
// Filter #0
// 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #1
// 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #2
// 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #3
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ...
// Filter #15
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #16
// 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #17
// 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #18
// 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Filter #19
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ...
mlir::Value DPUExpandRewriter::composeWeights(IE::ExpandOp origOp, const int64_t convolutionAlignment,
                                              mlir::PatternRewriter& rewriter) const {
    const auto origInShape = getShape(origOp.input());
    const auto origOutShape = getShape(origOp.output());
    const auto weightShape = calculateWeightsShape(origInShape, origOutShape, convolutionAlignment);
    const int64_t numInputChannels = origInShape[Dims4D::Act::C];
    const int64_t numOutputChannels = origOutShape[Dims4D::Act::C];
    const int64_t kernelOutputChannels = weightShape[Dims4D::Filter::OC];
    const int64_t kernelInputChannels = weightShape[Dims4D::Filter::IC];
    const int64_t kernelY = weightShape[Dims4D::Filter::KY];
    const int64_t kernelX = weightShape[Dims4D::Filter::KX];

    const auto strideIC = kernelY * kernelX;
    const auto strideOC = strideIC * kernelInputChannels;
    std::vector<ngraph::float16> weightValues(kernelOutputChannels * kernelInputChannels * kernelY * kernelX, 0.f);
    // Resulting weights can be split into the number of blocks defined by convolutionAlignment.
    // convolutionAlignment is usually 16. E.g. for 64x48x1x1 weights there are 4 blocks with 16x48x1x1 geometry.
    // Block offset iterates over these chunks. Each block must contain a diagonal matrix padded with zeros.
    // outChan iterates inside each chunk. The idea is to fill main diagonals with ones.
    // However, in order to maintain continuation, diagonal must be shifted by prefixSize.
    // We want to do this:
    // 1 0 0 0 0 0 0 0
    // 0 1 0 0 0 0 0 0
    // 0 0 1 0 0 0 0 0
    // 0 0 0 0 0 0 0 0
    // 0 0 0 1 0 0 0 0
    // Not this:
    // 1 0 0 0 0 0 0 0
    // 0 1 0 0 0 0 0 0
    // 0 0 1 0 0 0 0 0
    // 0 0 0 0 0 0 0 0
    // 1 0 0 0 0 0 0 0
    // Note that outChan iterates only over origInShape[Dims4D::Act::C] because we want to skip padded rows.
    // For the simple example counters go like that:
    //
    //           prefixSize (3)
    //                   |
    //             1 0 0 0 0 0 0 0
    //             0 1 0 0 0 0 0 0
    //             0 0 1 0 0 0 0 0
    //             0 0 0 0 0 0 0 0
    // blockRow -> 0 0 0 1 0 0 0 0
    //             ^     ^
    //       outChan     inChan = prefixSize + outChan (0 + 3)
    for (int64_t blockOffset = 0; blockOffset < convolutionAlignment; blockOffset++) {
        const auto blockRow = blockOffset * numOutputChannels;
        const auto prefixSize = blockOffset * numInputChannels;
        for (int64_t outChan = 0; outChan < numInputChannels; outChan++) {
            const auto inChan = prefixSize + outChan;
            const auto pos = (blockRow + outChan) * strideOC + inChan * strideIC;
            weightValues.at(pos) = 1.f;
        }
    }

    const auto ctx = rewriter.getContext();
    const auto weightType = mlir::RankedTensorType::get(weightShape.raw(), mlir::Float16Type::get(ctx));
    const auto weightAttr = mlir::DenseElementsAttr::get(weightType, makeArrayRef(weightValues));
    const auto weightContentAttr = Const::ContentAttr::get(weightAttr);
    const auto declLoc = appendLoc(origOp.getLoc(), "weights for DPU expand");
    auto declOp = rewriter.create<Const::DeclareOp>(declLoc, weightType, weightContentAttr);

    const auto reorderLoc = appendLoc(origOp.getLoc(), "reorder weights for DPU expand");
    const auto weightTypeNCHW = declOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto reorderType = weightTypeNCHW.changeDimsOrder(DimsOrder::NHWC);
    const auto orderMap = DimsOrder::NHWC.toAffineMap(ctx);
    auto reorderOut = rewriter.createOrFold<IE::ReorderOp>(reorderLoc, reorderType, declOp.output(), orderMap);

    return reorderOut;
}

IE::ConvolutionOp DPUExpandRewriter::buildConvolution(IE::ExpandOp expandOp, mlir::Value activation,
                                                      mlir::Value weights, mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    // IE::ConvolutionOp output type inference sets NCHW output order.
    // Specify convolution output type explicitly.
    const auto origOutType = expandOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto weightsShape = getShape(weights);
    const auto outChannels = weightsShape[Dims4D::Filter::OC];
    const Shape convInShape = getShape(activation).toValues();
    const Shape convOutShape = {convInShape[Dims4D::Act::N], outChannels, convInShape[Dims4D::Act::H],
                                convInShape[Dims4D::Act::W]};
    const auto convOutType = origOutType.changeShape(convOutShape);
    return rewriter.create<IE::ConvolutionOp>(expandOp.getLoc(), convOutType, activation, weights,
                                              /*bias=*/nullptr, strides, kernelPadsBegin, kernelPadsEnd, dilations,
                                              /*postOp=*/nullptr);
}

mlir::LogicalResult DPUExpandRewriter::matchAndRewrite(IE::ExpandOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.ExpandOp at '{1}'", getDebugName(), origOp->getLoc());
    if (!isEligibleForConversion(origOp)) {
        return matchFailed(rewriter, origOp, "[{0}] Cannot convert IE.ExpandOp at '{1}'", getDebugName(),
                           origOp->getLoc());
    }
    const auto expandInType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto convolutionAlignment = calculateAlignment(expandInType);
    auto reshapeIn = reshapeInput(origOp, convolutionAlignment, rewriter);
    auto weights = composeWeights(origOp, convolutionAlignment, rewriter);
    auto convOp = buildConvolution(origOp, reshapeIn.output(), weights, rewriter);
    auto reshapeOut = reshapeOutput(origOp, convOp.output(), rewriter);
    rewriter.replaceOp(origOp, reshapeOut.output());
    return mlir::success();
}

//
// ConvertExpandToConvPass
//

class ConvertExpandToConvPass final : public IE::ConvertExpandToConvPassBase<ConvertExpandToConvPass> {
public:
    explicit ConvertExpandToConvPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertExpandToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DPUExpandRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertExpandToConvPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertExpandToConvPass(Logger log) {
    return std::make_unique<ConvertExpandToConvPass>(log);
}
