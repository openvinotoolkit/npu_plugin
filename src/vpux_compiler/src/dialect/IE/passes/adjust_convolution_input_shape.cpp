//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"

#include <mlir/IR/IRMapping.h>

#include <openvino/core/validation_util.hpp>

using namespace vpux;

namespace {

// TODO: needs find suitable implict reshape value. Ticket: E#78751
constexpr int64_t CONVOLUTION_INPUT_SHAPE_ALIGNMENT = 4;

//
// ReshapeConvInput
//

template <typename ConcreteOp>
class ReshapeConvInput final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReshapeConvInput(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult ReshapeConvInput<ConcreteOp>::matchAndRewrite(ConcreteOp convOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    /*
        Convert 1x1 convolution from
            input          filter                    input               filter
        [1, C, H, 1]     [OC, C, 1, 1]             [1, C, H, 1]        [OC, C, 1, 1]
              \             /                =>        |                   |
                   Conv                            AffineReshape           |
               [1, OC, H, 1]                     [1, C, H/4, 4]            |
                                                       \                  /
                                                              Conv
                                                        [1, OC, H/4, 4]
                                                               |
                                                          AffineReshape
                                                          [1, OC, H, 1]
    */
    auto ctx = convOp->getContext();
    const auto inputShape = getShape(convOp.getInput());
    const auto filterShape = getShape(convOp.getFilter());

    // Current logic only works with input and filter shape with 4 dimensions
    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    // check suitable 1x1 convolution with input width = 1, strides = [1, 1]
    if (inputShape[Dims4D::Act::W] != 1 || filterShape[Dims4D::Filter::KX] != 1 ||
        filterShape[Dims4D::Filter::KY] != 1) {
        return mlir::failure();
    }

    const auto strides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    auto stridesEqualToOne = llvm::all_of(strides, [](const int64_t elem) {
        return elem == 1;
    });
    if (!stridesEqualToOne) {
        return mlir::failure();
    }

    if (inputShape[Dims4D::Act::H] % CONVOLUTION_INPUT_SHAPE_ALIGNMENT != 0) {
        return mlir::failure();
    }

    _log.trace("Adjust input shape for convolution at '{0}'", convOp->getLoc());
    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                             inputShape[Dims4D::Act::H] / CONVOLUTION_INPUT_SHAPE_ALIGNMENT,
                                             CONVOLUTION_INPUT_SHAPE_ALIGNMENT};

    const auto inputShapeAttr = getIntArrayAttr(convOp->getContext(), newInShape);
    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind()},
                                                   {Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                                                   {Dims4D::Act::W.ind()}};
    auto newInput = rewriter.create<IE::AffineReshapeOp>(convOp->getLoc(), convOp.getInput(),
                                                         getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);
    mlir::IRMapping mapper;
    mapper.map(convOp.getInput(), newInput.getOutput());
    auto newConvOp = mlir::dyn_cast<ConcreteOp>(rewriter.clone(*convOp, mapper));

    auto outputShape = getShape(convOp.getOutput());
    auto newOutputShape = Shape(SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C],
                                                     outputShape[Dims4D::Act::H] / CONVOLUTION_INPUT_SHAPE_ALIGNMENT,
                                                     outputShape[Dims4D::Act::W] * CONVOLUTION_INPUT_SHAPE_ALIGNMENT});

    auto newOutputType = newConvOp.getOutput().getType().template cast<vpux::NDTypeInterface>();
    newOutputType = newOutputType.changeShape(newOutputShape);
    newConvOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(newOutputType));
    const auto outShape = getShape(convOp.getOutput()).raw();
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);

    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(convOp, newConvOp.getOutput(),
                                                     getIntArrayOfArray(ctx, outDimMapping), outShapeAttr);

    return mlir::success();
}

//
// ReshapeExpandDWConvInput
//

class ReshapeExpandDWConvInput final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    ReshapeExpandDWConvInput(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReshapeExpandDWConvInput::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());

    // Only support GroupConvolution with constant filter
    // Kernel size must be 1x1, and must be a depthwise convolution.
    auto kernelShape = getShape(origOp.getFilter());
    if (kernelShape[Dims4D::Filter::KX] != 1 || kernelShape[Dims4D::Filter::KX] != 1 ||
        kernelShape[Dims4D::Filter::OC] != origOp.getGroups().value()) {
        return mlir::failure();
    }
    const auto logCb = [&](const formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };
    if (!VPU::NCEDepthConvolutionOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }
    // Check stride
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    if (strides[Dims4D::Strides::X.ind()] > 1 || strides[Dims4D::Strides::Y.ind()] == 1) {
        return mlir::failure();
    }
    auto parentExpandOp = origOp.getInput().getDefiningOp<IE::ExpandOp>();
    if (parentExpandOp == nullptr) {
        return mlir::failure();
    }
    if (!parentExpandOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    if (iface == nullptr) {
        return mlir::failure();
    }
    const auto alignment = iface.getInputChannelAlignment();

    const auto unExpandedInput = parentExpandOp.getInput();
    const auto unExpandedType = unExpandedInput.getType().cast<vpux::NDTypeInterface>();
    auto unExpandedShape = Shape(unExpandedType.getShape().toValues());

    auto IN = unExpandedShape[Dims4D::Act::N];
    auto IC = unExpandedShape[Dims4D::Act::C];
    auto IH = unExpandedShape[Dims4D::Act::H];
    auto IW = unExpandedShape[Dims4D::Act::W];

    if (IC % alignment == 0) {
        _log.trace("Channel is already aligned");
        return mlir::failure();
    }
    // Check if can align
    if (IC * IW % alignment != 0) {
        _log.trace("Channel cannot be aligned");
        return mlir::failure();
    }

    auto constInput = origOp.getFilter().getDefiningOp<Const::DeclareOp>();
    auto realDataSizeResult = vpux::IE::getBaseContentNumElements(constInput);
    auto activationDataSize =
            std::accumulate(unExpandedShape.begin(), unExpandedShape.end(), int64_t(1), std::multiplies<int64_t>());
    if (mlir::failed(realDataSizeResult) ||
        (realDataSizeResult.value() != 1 && realDataSizeResult.value() != activationDataSize)) {
        _log.trace("Unsupported const input {0} at {1}", constInput->getName(), constInput->getLoc());
        return mlir::failure();
    }

    // Insert ShapeCast to align input shape
    // For example: 1x3x640x640 -> 1x48x640x40
    auto newIC = alignment * IC;
    auto newIW = IC * IW / newIC;
    auto alignedInShape = Shape({IN, newIC, IH, newIW});
    auto alignedInputShapeAttr = getIntArrayAttr(rewriter.getContext(), alignedInShape);
    const auto dstType = unExpandedType.changeShape(ShapeRef(alignedInShape));

    auto shapeCastInputOp =
            rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), dstType, unExpandedInput, alignedInputShapeAttr);

    auto contentAttr = constInput.getContentAttr();
    auto baseContent = contentAttr.getBaseContent();
    auto dataShape = getShape(constInput.getOutput()).toValues();

    Const::ContentAttr newContentAttr = Const::ContentAttr::get(baseContent);
    Shape realDataShape = baseContent.getShapedType().getShape();

    auto newConstOutputType = constInput.getOutput().getType().cast<vpux::NDTypeInterface>();
    newContentAttr = newContentAttr.broadcast(Dims4D::Act::N, alignedInShape[Dims4D::Act::C]);
    auto newConstantShape = Shape(newConstOutputType.getShape().size(), int64_t(1));
    newConstantShape[Dims4D::Act::N] = alignedInShape[Dims4D::Act::C];
    newConstOutputType = newConstOutputType.changeShape(newConstantShape);
    newContentAttr = newContentAttr.reshape(newConstantShape);

    for (auto& attr : contentAttr.getTransformations()) {
        if (attr.isa<Const::PadWithZeroAttr>() || attr.isa<Const::BroadcastAttr>()) {
            // skip the attributes that the contentAttr already contains
            continue;
        }
        if (attr.isa<Const::ReshapeAttr>()) {
            // Only remain the reshape attribute when it's used for dimension expansion to 4D,
            // and for dimension shrink from 5D to 4D
            // e.g., from [1x512] to [1x1x1x512]
            auto reshapeAttr = attr.cast<Const::ReshapeAttr>();
            auto reshapeShape = Shape(parseIntArrayAttr<int64_t>(reshapeAttr.getShape()));
            if (vpux::IE::isNotDimExpansionReshape(realDataShape, reshapeShape) &&
                vpux::IE::isNotDimShrinkReshape(realDataShape, reshapeShape)) {
                continue;
            }
        }
        newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
    }

    auto newConstInput = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newConstOutputType, newContentAttr);

    // Infer group conv output shape
    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    auto convInShape = to_small_vector(alignedInShape.raw());
    convInShape[1] /= newIC;
    auto filterShape = to_small_vector(newConstOutputType.getShape().raw());

    const auto outputShape = ov::infer_convolution_forward(
            EmptyNode::instance(), ov::Shape(convInShape.begin(), convInShape.end()),
            ov::Strides(windowStrides.size(), 1),  // dummy data dilations
            ov::CoordinateDiff(dataPaddingBelow.begin(), dataPaddingBelow.end()),
            ov::CoordinateDiff(dataPaddingAbove.begin(), dataPaddingAbove.end()),
            ov::Shape(filterShape.begin(), filterShape.end()), ov::Strides(windowStrides.begin(), windowStrides.end()),
            ov::Strides(windowDilations.begin(), windowDilations.end()));
    const auto shapeI64 = to_small_vector(outputShape.get_shape() | transformed([](size_t val) {
                                              return checked_cast<int64_t>(val);
                                          }));
    const auto origOutType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto convOutType = unExpandedType.changeShapeElemType(Shape(shapeI64), origOutType);
    auto groupsAttr = getIntAttr(rewriter, newIC);
    auto grpConv = rewriter.create<IE::GroupConvolutionOp>(
            origOp->getLoc(), convOutType, shapeCastInputOp, newConstInput, origOp.getBias(), origOp.getStridesAttr(),
            origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilationsAttr(), groupsAttr,
            /*post_opAttr=*/nullptr, /*clamp=*/nullptr);

    // Insert ShapeCast to reshape the output to original outShape
    auto unExpandedOutShape = Shape({IN, IC, convOutType.getShape()[Dims4D::Act::H], IW});
    auto shapeCastOutputAttr = getIntArrayAttr(rewriter.getContext(), unExpandedOutShape);
    auto shapeCastOutputOp = rewriter.create<IE::ShapeCastOp>(
            origOp->getLoc(), convOutType.changeShape(unExpandedOutShape), grpConv, shapeCastOutputAttr);

    auto newOutputExpandOp = rewriter.create<IE::ExpandOp>(
            origOp->getLoc(), shapeCastOutputOp, parentExpandOp.getPadsBeginAttr(), parentExpandOp.getPadsEndAttr());

    // Replace with new sub graph
    rewriter.replaceOp(origOp, newOutputExpandOp->getResult(0));

    return mlir::success();
}

//
// AdjustConvolutionInputShape
//

class AdjustConvolutionInputShapePass final :
        public IE::AdjustConvolutionInputShapeBase<AdjustConvolutionInputShapePass> {
public:
    explicit AdjustConvolutionInputShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionInputShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // Adjust between H and W like [1, C, H, W]  -> [1, C, H/4, W*4]
    patterns.add<ReshapeConvInput<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ReshapeConvInput<IE::GroupConvolutionOp>>(&ctx, _log);

    // Adust between C and H/W like [1, C, H, W] -> [1, C*4, H, W/4]
    // Also need stride[H] > 1
    patterns.add<ReshapeExpandDWConvInput>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertFCToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionInputShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionInputShapePass>(log);
}
