//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {

//
// FoldConvStrideKernel
//

class FoldConvStrideKernel final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FoldConvStrideKernel(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// This pass want to fold the Convolution's stride attribute to 1 in DimW
//  through adjust the input shape and kernel shape.
//  In this way, it will decrease the expand channels to decrease DMA data copy
//  For example:
//          N  H   W  C             N  H   W  C
//    Input 1x128x128x8             1x128x64x16
//          OC Y X IC       =>     OC Y X IC
//    Kernel 2x1x2x8                2x1x1x16
//    Stride   1 2                    1 1
//  In the ExpandActivation pass, it doesn't need expand the input channel
//
mlir::LogicalResult FoldConvStrideKernel::matchAndRewrite(IE::ConvolutionOp convOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto strides = Shape(parseIntArrayAttr<int64_t>(convOp.getStrides()));
    auto filter = convOp.getFilter();
    // Don't need to consider bias, the function not change the output shape.

    auto inNDInterface = convOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto inDimOrder = inNDInterface.getDimsOrder();
    if (DimsOrder::NHWC != inDimOrder) {
        return mlir::failure();
    }

    auto filterConst = filter.getDefiningOp<Const::DeclareOp>();
    if (filterConst == nullptr) {
        return mlir::failure();
    }

    auto filterShape = vpux::getShape(filter);
    if ((1 == strides[Dims4D::Strides::X]) || (filterShape[Dims4D::Filter::KX] > strides[Dims4D::Strides::X])) {
        return mlir::failure();
    }
    auto inputShape = inNDInterface.getShape();
    if (inputShape[Dims4D::Act::W] % strides[Dims4D::Strides::X]) {
        return mlir::failure();
    }

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedChannel = iface.getInputChannelAlignment();
    if ((filterShape[Dims4D::Filter::IC] % alignedChannel) == 0) {
        // Already aligned
        return mlir::failure();
    }
    Shape newShape(inputShape.raw());
    newShape[Dims4D::Act::W] /= strides[Dims4D::Strides::X];
    newShape[Dims4D::Act::C] *= strides[Dims4D::Strides::X];
    const auto ctx = rewriter.getContext();
    const auto dstType = inNDInterface.changeShape(newShape);
    const auto targetShapeAttr = getIntArrayAttr(ctx, newShape.raw());
    auto inputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(convOp.getLoc(), dstType, convOp.getInput(), targetShapeAttr);

    Shape newFilterShape(filterShape.raw());
    newFilterShape[Dims4D::Filter::IC] *= newFilterShape[Dims4D::Filter::KX];
    newFilterShape[Dims4D::Filter::KX] = 1;
    auto cstContentAttrFilter = filterConst.getContentAttr();
    cstContentAttrFilter = cstContentAttrFilter.reshape(newFilterShape);
    if (newShape[Dims4D::Act::C] != newFilterShape[Dims4D::Filter::IC]) {
        int64_t padding = newShape[Dims4D::Act::C] - newFilterShape[Dims4D::Filter::IC];
        cstContentAttrFilter = cstContentAttrFilter.padWithZero({0, 0, 0, 0}, {0, padding, 0, 0});
    }
    auto newFilter =
            rewriter.create<Const::DeclareOp>(convOp.getLoc(), cstContentAttrFilter.getType(), cstContentAttrFilter);

    auto newStride = strides;
    newStride[Dims4D::Strides::X] = 1;
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            convOp, convOp.getType(), inputShapeCastOp, newFilter, convOp.getBias(),
            getIntArrayAttr(ctx, newStride.raw()), convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(),
            convOp.getDilationsAttr(), convOp.getPostOpAttr(), convOp.getClampAttr());
    return mlir::success();
}

//
// AdjustConvShape
//

class AdjustConvShape final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    AdjustConvShape(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value reshapeBias(mlir::PatternRewriter& rewriter, mlir::Value bias, ShapeRef outShape) {
    if (bias == nullptr) {
        return nullptr;
    }
    auto cst = bias.getDefiningOp<Const::DeclareOp>();
    auto biasShape = getShape(bias);
    auto biasCxW = biasShape[Dims4D::Act::C] * biasShape[Dims4D::Act::W];
    auto outCxW = outShape[Dims4D::Act::C] * outShape[Dims4D::Act::W];
    if (biasCxW == 1) {
        return bias;
    }
    auto contentAttr = cst.getContentAttr();
    Shape newOutSahpe(biasShape.raw());
    newOutSahpe[Dims4D::Act::C] = outShape[Dims4D::Act::C];
    newOutSahpe[Dims4D::Act::W] = outShape[Dims4D::Act::W];
    if (biasCxW != outCxW) {
        auto dimValue = outShape[Dims4D::Act::C];
        auto broadCastDim = Dims4D::Act::C;
        if (outShape[Dims4D::Act::C] % biasShape[Dims4D::Act::C]) {
            dimValue = outCxW / biasShape[Dims4D::Act::C];
            broadCastDim = Dims4D::Act::W;
        } else {
            newOutSahpe[Dims4D::Act::W] = biasShape[Dims4D::Act::W];
        }
        contentAttr = contentAttr.broadcast(broadCastDim, dimValue);
    }
    contentAttr = contentAttr.reshape(newOutSahpe);
    return rewriter.create<Const::DeclareOp>(bias.getLoc(), contentAttr.getType(), contentAttr);
}

//
// Avoid expand though adjust the Convolution's Shape
// For example:
//          N  H  W C       N  H  W C
//   Input  1 16 16 3 -+
//                     |-> 1 16 16 3
//   Kernel 3  1  1 3 -+
//             |
//             V
//          N  H  W C        N  H  W C     N  H  W C
//   Input  1 16  1 48 -+
//                      |->  1 16  1 48 -> 1 16 16 3
//   Kernel 48 1  1 48 -+
//
mlir::LogicalResult AdjustConvShape::matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), convOp->getName(), convOp->getLoc());

    auto inNDInterface = convOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto inDimOrder = inNDInterface.getDimsOrder();
    auto outNDInterface = convOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    auto outDimOrder = outNDInterface.getDimsOrder();
    if (DimsOrder::NHWC != inDimOrder || DimsOrder::NHWC != outDimOrder) {
        _log.trace("The input/output layout should be NHWC, but got {0}/{1}", inDimOrder, outDimOrder);
        return mlir::failure();
    }

    auto isQuantizedType = [](NDTypeInterface ndType) {
        const auto elementType = ndType.getElementType();
        return mlir::isa<mlir::quant::QuantizedType>(elementType);
    };

    auto filter = convOp.getFilter();
    auto filterNDInterface = filter.getType().dyn_cast<vpux::NDTypeInterface>();
    if (isQuantizedType(inNDInterface) || isQuantizedType(filterNDInterface) || isQuantizedType(outNDInterface)) {
        _log.trace("Unsupported Convolution with Quantized Type");
        return mlir::failure();
    }

    auto filterShape = vpux::getShape(filter);
    auto isConst = [](mlir::Value value) {
        auto cst = value.getDefiningOp<Const::DeclareOp>();
        if (nullptr == cst) {
            return false;
        }
        return true;
    };

    auto bias = convOp.getBias();
    if (!isConst(filter) || (bias && !isConst(bias))) {
        _log.trace("Unsupported filter and bias of Convolution is not Constant");
        return mlir::failure();
    }
    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedInputChannel = iface.getInputChannelAlignment();
    int64_t alignedOutputChannel = iface.getOutputChannelAlignment();
    auto caculateExpandShapeSize = [](ShapeRef shape, int64_t alignedChannel) {
        auto expandShape = shape.toValues();
        expandShape[Dims4D::Act::C] = alignValUp(shape[Dims4D::Act::C], alignedChannel);
        return expandShape.totalSize();
    };

    if ((filterShape[Dims4D::Filter::IC] % alignedInputChannel) == 0 &&
        (filterShape[Dims4D::Filter::OC] % alignedOutputChannel) == 0) {
        _log.trace("The input/output channel aready aligend");
        return mlir::failure();
    }

    auto inputShape = inNDInterface.getShape();
    auto outputShape = outNDInterface.getShape();

    auto wcInDimSize = inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::W];
    if (wcInDimSize % alignedInputChannel) {
        _log.trace("The input channel*width ({0}) can't get align factor {1}", wcInDimSize, alignedInputChannel);
        return mlir::failure();
    }

    auto wcOutDimSize = outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::W];
    if (wcOutDimSize % alignedOutputChannel) {
        // We want output channel align to input channel because the compressed CONV's input channel alignment is 4
        // If the output channel is not multiple of alignedInputChannel, it will not work.
        if ((wcOutDimSize % alignedInputChannel) || (alignedOutputChannel % alignedInputChannel)) {
            _log.trace("The output channel*width ({0}) can't get align factor {1}", wcOutDimSize, alignedOutputChannel);
            return mlir::failure();
        }
        alignedOutputChannel = alignedInputChannel;
    }

    const auto ctx = rewriter.getContext();

    auto calcBorrowFactor = [](int64_t channel, int64_t alignedChannel) {
        auto leastAlignedChannel = std::lcm(channel, alignedChannel);
        return (leastAlignedChannel / channel);
    };

    auto borrowIn = calcBorrowFactor(inputShape[Dims4D::Act::C], alignedInputChannel);
    auto borrowOut = calcBorrowFactor(outputShape[Dims4D::Act::C], alignedOutputChannel);
    _log.trace("Input factor {0}, output factor {1}", borrowIn, borrowOut);

    //
    // Promise the input channel align first
    // To reshape the input tensor from DimW, we need promise the new shape's next W index is (stride*N).
    // Because the new convolution's stride is 1.
    //
    const auto strides = Shape(parseIntArrayAttr<int64_t>(convOp.getStrides()));
    auto realInFactor = std::lcm(strides[Dims4D::Strides::X], borrowIn);

    if (inputShape[Dims4D::Act::W] % realInFactor) {
        _log.info("Don't have factor {0} in input DimW", realInFactor);
        return mlir::failure();
    }

    if (outputShape[Dims4D::Act::W] % borrowOut) {
        _log.info("Don't have factor {0} in output DimW", borrowOut);
        return mlir::failure();
    }

    // To promise the new kernel's IC >= originIC * originKX
    //  And the MAX realInFactor is inputShape[Dims4D::Act::W]
    auto newInputDimW = inputShape[Dims4D::Act::W] / realInFactor;
    while (realInFactor < filterShape[Dims4D::Filter::KX] && newInputDimW > 1) {
        auto divisor = vpux::smallestDivisor(newInputDimW);
        realInFactor *= divisor;
        newInputDimW /= divisor;
    }

    auto padBegin = Shape(parseIntArrayAttr<int64_t>(convOp.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(convOp.getPadsEnd()));

    int64_t leftPading = 0;
    Shape newInputShape(inputShape.raw());
    Shape newOutputShape(outputShape.raw());
    Shape newFilterShape(filterShape.raw());
    int64_t borrowFactor;
    if (filterShape[Dims4D::Filter::KX] == 1) {
        // If KX = 1, the DimC can borrow any dims from DimW
        // Special case to make the kernel size as small as possible
        borrowFactor = std::max(borrowIn, borrowOut);
        newInputShape[Dims4D::Act::W] /= borrowFactor;
        newInputShape[Dims4D::Act::C] *= borrowFactor;

        newFilterShape[Dims4D::Filter::IC] *= borrowFactor;
        newFilterShape[Dims4D::Filter::OC] *= borrowFactor;
        newOutputShape[Dims4D::Act::W] /= borrowFactor;

        leftPading -= (padBegin[Dims4D::PadsBegin::Left] * filterShape[Dims4D::Filter::IC]);
    } else {
        borrowFactor = realInFactor / strides[Dims4D::Strides::X];
        if (borrowFactor < borrowOut) {
            // Output channel not aligned and check Input can borrow
            // If can, allocate new channels
            // If not, let input channel align
            auto outBorrowFact = std::lcm(borrowFactor, borrowOut);
            if ((newInputShape[Dims4D::Act::W] % (outBorrowFact * strides[Dims4D::Strides::X])) == 0 &&
                ((outputShape[Dims4D::Act::W] % outBorrowFact) == 0)) {
                borrowFactor = outBorrowFact;
                realInFactor = borrowFactor * strides[Dims4D::Strides::X];
            }
        }

        if (outputShape[Dims4D::Act::W] % borrowFactor) {
            _log.info("The outputShape not aligned");
            return mlir::failure();
        }

        newInputShape[Dims4D::Act::W] /= realInFactor;
        newInputShape[Dims4D::Act::C] *= realInFactor;
        //
        // The newFilterIC >= originFilterKX * originFilterIC and newFilterIC = N * stride
        // Generally, the newKX = 2 is enough to cover full origin's calculation.
        // For example:
        //          N H W C
        //   Input: 1x4x4x3
        //  Filter: 1x3x3x3
        //  Stride:   1x2
        //  If we borrow factor 4 from W
        //    NewFilter: 2x3x2x12
        // OC = 0
        //  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
        //   c11 c12 c13 c21 c22 c23 c31 c32 c33  0    0    0
        //   0    0   0   0   0   0   0   0   0   0    0    0
        // OC = 1
        //  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
        //   0    0   0   0   0   0  c11 c12 c13  c21  c22  c23
        //   c31 c32 c33  0   0   0   0   0   0   0    0    0
        //
        // When consider the left and right padding, it need add another kernel to make it work.
        //
        if (padBegin[Dims4D::PadsBegin::Left] == 0 || padEnd[Dims4D::PadsEnd::Right] == 0) {
            newFilterShape[Dims4D::Filter::KX] = 2;
        } else {
            newFilterShape[Dims4D::Filter::KX] = 3;
        }

        if (padBegin[Dims4D::PadsBegin::Left] > 0) {
            leftPading = (realInFactor - padBegin[Dims4D::PadsBegin::Left]) * filterShape[Dims4D::Filter::IC];
        }

        newFilterShape[Dims4D::Filter::IC] = newInputShape[Dims4D::Act::C];
        newFilterShape[Dims4D::Filter::OC] *= borrowFactor;

        newOutputShape[Dims4D::Act::W] /= borrowFactor;
    }
    newOutputShape[Dims4D::Act::C] = newFilterShape[Dims4D::Filter::OC];
    auto newFilterSize = newFilterShape.totalSize();
    _log.trace("The new shape {0}, new filter shape {1}, filter size {2}", newInputShape, newFilterShape,
               newFilterSize);

    auto expandedInputSize = caculateExpandShapeSize(inputShape, alignedInputChannel);
    auto expandedOutputSize = caculateExpandShapeSize(outputShape, alignedOutputChannel);
    auto expandedTotalSize = expandedInputSize + expandedOutputSize;
    const auto cmxMemSize = VPU::getTotalCMXSize(convOp.getOperation());
    const auto elementSize = inNDInterface.getCompactAllocSize().count() / inputShape.totalSize();

    if (expandedTotalSize * elementSize < cmxMemSize.count() || newFilterSize > Byte(1_MB).count()) {
        return mlir::failure();
    }

    // As input channel already aligned, output channel unaligned, it only need slice the data.
    // So filter out the wasted calculation greater than slice
    auto kernelScaled = static_cast<float>(newFilterShape.totalSize()) / static_cast<float>(filterShape.totalSize());
    auto outputTensorScaled = static_cast<float>(expandedOutputSize) / static_cast<float>(outputShape.totalSize());
    if ((filterShape[Dims4D::Filter::IC] % alignedInputChannel) == 0 &&
        (kernelScaled / outputTensorScaled) > alignedOutputChannel) {
        _log.trace("The shape adjust cost greater than expand when input channel already aligned");
        return mlir::failure();
    }

    auto newFilterICxKX = newFilterShape[Dims4D::Filter::IC] * newFilterShape[Dims4D::Filter::KX];
    auto oldFilterICxKX = filterShape[Dims4D::Filter::IC] * filterShape[Dims4D::Filter::KX];
    Shape middleFilterShape = {filterShape[Dims4D::Filter::OC], oldFilterICxKX, 1, filterShape[Dims4D::Filter::KY]};
    auto cstContentAttrFilter = filter.getDefiningOp<Const::DeclareOp>().getContentAttr();
    const auto totalPading = newFilterICxKX - oldFilterICxKX;
    SmallVector<mlir::Value> filterConst;
    //
    // Construct the new filter
    // For a NHWC layout Conv:
    //        N  H  W C       N  H  W C
    // Input  1 16 16 3 -+
    //                   |-> 1 16 16 3
    // Kernel 3  1  1 3 -+
    //           |
    //           V
    //        N  H  W C        N  H  W C     N  H  W C
    // Input  1 16  1 48 -+
    //                    |-> 1 16  1 48 -> 1 16 16 3
    // Kernel 48 1  1 48 -+
    //
    // The borrowFactor = 16
    // The new kernel:
    //   Padding 0 in input channel to (3x1x1x48)
    //   Concat in output channel to (48x1x1x48)
    //
    for (int64_t i = 0; i < borrowFactor; i++) {
        auto newCstContent = cstContentAttrFilter.reshape(middleFilterShape);
        auto newLeftPading = (leftPading > 0) ? leftPading : 0;
        auto newRightPading = (totalPading > leftPading) ? (totalPading - leftPading) : 0;
        Shape cstPadBegin = {0, newLeftPading, 0, 0};
        Shape cstPadEnd = {0, newRightPading, 0, 0};
        newCstContent = newCstContent.padWithZero(cstPadBegin, cstPadEnd);
        if (newLeftPading + newRightPading > totalPading) {
            Shape offset = {0, (leftPading > 0) ? 0 : -leftPading, 0, 0};
            Shape viewShape(middleFilterShape.raw());
            viewShape[Dims4D::Filter::IC] += totalPading;
            newCstContent = newCstContent.subview(offset, viewShape);
        }
        auto temp = rewriter.create<Const::DeclareOp>(convOp.getLoc(), newCstContent.getType(), newCstContent);
        filterConst.push_back(temp);
        leftPading += filterShape[Dims4D::Filter::IC] * strides[Dims4D::Strides::X];
    }
    auto newFilterConcatOp = rewriter.create<IE::ConcatOp>(convOp.getLoc(), filterConst, Dims4D::Filter::OC);
    auto newFilterType = filter.getType().dyn_cast<vpux::NDTypeInterface>().changeShape(newFilterShape);
    auto newFilter = rewriter.create<IE::ShapeCastOp>(convOp.getLoc(), newFilterType, newFilterConcatOp.getOutput(),
                                                      getIntArrayAttr(ctx, newFilterShape.raw()));

    // Pading on the Dim W already handled by the const construct
    auto newBeginAttr = convOp.getPadsBeginAttr();
    auto padBVect = parseIntArrayAttr<int64_t>(newBeginAttr);
    padBVect[Dims4D::PadsBegin::Left.ind()] = padBegin[Dims4D::PadsBegin::Left] > 0 ? 1 : 0;

    auto newEndAttr = convOp.getPadsEndAttr();
    auto padEVect = parseIntArrayAttr<int64_t>(newEndAttr);
    padEVect[Dims4D::PadsEnd::Right.ind()] = padEnd[Dims4D::PadsEnd::Right] > 0 ? 1 : 0;

    // New Stride
    auto newStride = strides;
    newStride[Dims4D::Strides::X] = 1;

    auto newBias = reshapeBias(rewriter, convOp.getBias(), newOutputShape);

    const auto dstType = inNDInterface.changeShape(newInputShape);
    const auto targetShapeAttr = getIntArrayAttr(ctx, newInputShape.raw());
    auto inputShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(convOp.getLoc(), dstType, convOp.getInput(), targetShapeAttr);

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(convOp.getLoc(), inputShapeCastOp, newFilter, newBias,
                                                        getIntArrayAttr(ctx, newStride), getIntArrayAttr(ctx, padBVect),
                                                        getIntArrayAttr(ctx, padEVect), convOp.getDilationsAttr(),
                                                        convOp.getPostOpAttr(), convOp.getClampAttr());
    changeDimsOrder(newConvOp, outDimOrder, _log.nest());
    const auto outShapeAttr = getIntArrayAttr(ctx, outNDInterface.getShape().raw());
    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(convOp, outNDInterface, newConvOp.getOutput(), outShapeAttr);
    return mlir::success();
}

//
// AdjustConvolutionShapePass
//

class AdjustConvolutionShapePass final : public IE::AdjustConvolutionShapeBase<AdjustConvolutionShapePass> {
public:
    explicit AdjustConvolutionShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldConvStrideKernel>(&ctx, _log);
    patterns.add<AdjustConvShape>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAdjustConvolutionShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionShapePass>(log);
}
