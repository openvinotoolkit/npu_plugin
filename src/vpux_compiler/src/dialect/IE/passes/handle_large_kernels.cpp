//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/loop.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// GeneralPoolingBaseRewriter
//

template <class ConcreteOp>
class GeneralPoolingBaseRewriter : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GeneralPoolingBaseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("GeneralPoolingBaseRewriter");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

    virtual bool isLegalGeneralPoolingOp(ConcreteOp origOp) const final;
    virtual mlir::Value createFirstSplitOp(ConcreteOp origOp, IE::KernelsInfo& kernelsInfoVal,
                                           mlir::PatternRewriter& rewriter) const final;
    virtual mlir::Value createSecondSplitOp(ConcreteOp origOp, mlir::Value input, IE::KernelsInfo& kernelsInfoVal,
                                            mlir::PatternRewriter& rewriter) const final;
    virtual mlir::Value mappingNewOp(ConcreteOp origOp, mlir::Value input, mlir::ArrayAttr kernelsAttr,
                                     mlir::ArrayAttr padBeginAttr, mlir::ArrayAttr padEndAttr,
                                     mlir::ArrayAttr stridesAttr, mlir::PatternRewriter& rewriter) const final;

    virtual mlir::Value handlePoolWithPadding(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const = 0;

protected:
    Logger _log;
};

// General Pooling Op is global pooling or the kernel size equal stride size.
// - Global pooling case:
//   Pooling Op with kernel [24, 24], pads_begin = [0, 0], pads_end = [0, 0], strides = [24, 24]
//   Input shape is 1x16x24x24 and the output shape will be 1x16x1x1
// - Kernel size equal stride size case:
//   Pooling Op with kernel [24, 24], pads_begin = [0, 0], pads_end = [0, 0], strides = [24, 24]
//   Input shape is 1x16x96x96 and the output shape will be 1x16x4x4
// For those scenarios, MaxPool and AvgPool have the same split logic.
template <class ConcreteOp>
bool GeneralPoolingBaseRewriter<ConcreteOp>::isLegalGeneralPoolingOp(ConcreteOp origOp) const {
    const auto inDataType = origOp.input().getType().template cast<vpux::NDTypeInterface>();
    const auto inDataShape = inDataType.getShape().raw();
    if (inDataShape.size() != 4) {
        _log.trace("Pooling Op should with input shape rank equal 4");
        return true;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    if (vpux::IE::hasSupportedKernels(Shape(kernelSize))) {
        _log.trace("Kernel size of general Pooling Op is legal");
        return true;
    }

    // VPUX3700 does not support HW AvgPool
    // Experiments show kernel with one axis equal to 1 will get more performance
    const auto arch = VPU::getArch(origOp);
    if (mlir::isa<IE::AvgPoolOp>(origOp) && arch == VPU::ArchKind::VPUX30XX &&
        (kernelSize[Dims4D::Kernel::X.ind()] == 1 || kernelSize[Dims4D::Kernel::Y.ind()] == 1)) {
        _log.trace("AvgPool operation ignored by HandleLargeKernel pass for performance");
        return true;
    }

    // Split large kernel for genaral pooling Op has two limitations:
    // - The Pad attr should be all zero.
    // - Should not has data overlapped. It should be global pooling or stride equal to kernel size.
    const auto isNotZero = [](const auto& padVal) {
        return padVal != 0;
    };
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    if (llvm::any_of(padsBegin, isNotZero) || llvm::any_of(padsEnd, isNotZero)) {
        _log.trace("Pad attr of general Pooling Op should be all zero");
        return true;
    }

    auto isDataHasOverlapped = [&](int32_t kernelIdx, int32_t actIdx, int32_t strideIdx) {
        return (kernelSize[kernelIdx] < inDataShape[actIdx] && kernelSize[kernelIdx] != strides[strideIdx]);
    };
    if (isDataHasOverlapped(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind()) ||
        isDataHasOverlapped(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
        _log.trace("General Pooling Op should with global pooling or stride equal to kernel size");
        return true;
    }

    return false;
}

template <class ConcreteOp>
mlir::Value GeneralPoolingBaseRewriter<ConcreteOp>::mappingNewOp(
        ConcreteOp origOp, mlir::Value input, mlir::ArrayAttr kernelsAttr, mlir::ArrayAttr padBeginAttr,
        mlir::ArrayAttr padEndAttr, mlir::ArrayAttr stridesAttr, mlir::PatternRewriter& rewriter) const {
    mlir::BlockAndValueMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{input});
    auto* newPoolingOp = rewriter.clone(*origOp.getOperation(), mapper);

    VPUX_THROW_UNLESS(newPoolingOp->hasAttr("pads_begin") && newPoolingOp->hasAttr("pads_end") &&
                              newPoolingOp->hasAttr("strides") && newPoolingOp->hasAttr("kernel_size"),
                      "Cannot get all attributions");
    newPoolingOp->setAttr("kernel_size", kernelsAttr);
    newPoolingOp->setAttr("pads_begin", padBeginAttr);
    newPoolingOp->setAttr("pads_end", padEndAttr);
    newPoolingOp->setAttr("strides", stridesAttr);
    vpux::inferReturnTypes(newPoolingOp, vpux::InferShapedTypeMode::ALL);

    return newPoolingOp->getResult(0);
}

template <class ConcreteOp>
mlir::Value GeneralPoolingBaseRewriter<ConcreteOp>::createFirstSplitOp(ConcreteOp origOp, IE::KernelsInfo& kernelsInfo,
                                                                       mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();
    const auto firstOpKernelAttr = getIntArrayAttr(ctx, kernelsInfo.firstKernel);
    const auto firstOpPadBeginAttr = getIntArrayAttr(ctx, kernelsInfo.padBegin);
    const auto firstOpPadEndAttr = getIntArrayAttr(ctx, kernelsInfo.padEnd);

    // If it is global pooling, set the stride with 1
    // Elsewise set stride equal to kernel size
    auto getStrides = [](ShapeRef inShape, ShapeRef kernel, ShapeRef padBegin, ShapeRef padEnd) {
        auto strides = Shape{1, 1};

        if (kernel[Dims4D::Kernel::Y] !=
            (inShape[Dims4D::Act::H] + padBegin[Dims4D::PadsBegin::Top] + padEnd[Dims4D::PadsEnd::Bottom]))
            strides[Dims4D::Strides::Y] = kernel[Dims4D::Strides::Y];

        if (kernel[Dims4D::Kernel::X] !=
            (inShape[Dims4D::Act::W] + padBegin[Dims4D::PadsBegin::Left] + padEnd[Dims4D::PadsEnd::Right]))
            strides[Dims4D::Strides::X] = kernel[Dims4D::Strides::X];

        return strides;
    };

    const auto firstOpStrideAttr = getIntArrayAttr(ctx, getStrides(getShape(origOp.input()), kernelsInfo.firstKernel,
                                                                   kernelsInfo.padBegin, kernelsInfo.padEnd));

    const auto firstOpOutput = mappingNewOp(origOp, origOp.input(), firstOpKernelAttr, firstOpPadBeginAttr,
                                            firstOpPadEndAttr, firstOpStrideAttr, rewriter);

    _log.trace("Create firstSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
               firstOpKernelAttr, firstOpStrideAttr, firstOpPadBeginAttr, firstOpPadEndAttr);
    return firstOpOutput;
}

template <class ConcreteOp>
mlir::Value GeneralPoolingBaseRewriter<ConcreteOp>::createSecondSplitOp(ConcreteOp origOp, mlir::Value input,
                                                                        IE::KernelsInfo& kernelsInfo,
                                                                        mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();
    const auto secondOpInShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto secondOpKernel = kernelsInfo.secondKernel;
    auto globalAvgOverH = secondOpInShape[Dims4D::Act::H] == secondOpKernel[Dims4D::Kernel::Y];
    auto globalAvgOverW = secondOpInShape[Dims4D::Act::W] == secondOpKernel[Dims4D::Kernel::X];
    auto secondOpStride = Shape{globalAvgOverH ? 1 : secondOpKernel[Dims4D::Kernel::Y],
                                globalAvgOverW ? 1 : secondOpKernel[Dims4D::Kernel::X]};

    const auto secondOpKernelAttr = getIntArrayAttr(ctx, secondOpKernel);
    const auto secondOpStridesAttr = getIntArrayAttr(ctx, secondOpStride);
    const auto secondOpPadBeginAttr = getIntArrayAttr(ctx, makeArrayRef({0, 0}));
    const auto secondOpPadEndAttr = getIntArrayAttr(ctx, makeArrayRef({0, 0}));

    const auto secondOpOutput = mappingNewOp(origOp, input, secondOpKernelAttr, secondOpPadBeginAttr,
                                             secondOpPadEndAttr, secondOpStridesAttr, rewriter);

    _log.trace("Create secondSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
               secondOpKernelAttr, secondOpStridesAttr, secondOpPadBeginAttr, secondOpPadEndAttr);
    return secondOpOutput;
}

template <class ConcreteOp>
mlir::LogicalResult GeneralPoolingBaseRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' layer at '{2}'", this->getDebugName(), origOp.getOperationName(), origOp->getLoc());

    if (isLegalGeneralPoolingOp(origOp)) {
        return mlir::failure();
    }

    const auto origKernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    auto kernelsInfo = IE::calculateKernelsInfo(Shape(origKernel), _log.nest(2));
    if (!kernelsInfo.has_value()) {
        _log.trace("[{0}] Cannot spilt large kernel of '{1}' layer at '{2}'", this->getDebugName(),
                   origOp.getOperationName(), origOp->getLoc());
        return mlir::failure();
    }
    auto kernelsInfoVal = kernelsInfo.value();

    // Step 1: Handle the first split Op
    auto firstOpOutput = createFirstSplitOp(origOp, kernelsInfoVal, rewriter);

    // Step 2: Handle padding to avoid accuracy issue
    // Some scenarios need pad for first split Op.
    // For example: Pooling Op with kernel [23, 23], pads_begin = [0, 0], pads_end = [0, 0], strides = [23, 23]
    // - Split 1: Kernel: [8, 8], Stride: [8, 8], PadBegin: [0, 0], PadEnd: [1, 1] (need padding)
    // - Split 2: Kernel: [3, 3], Stride: [1, 1], PadBegin: [0, 0], PadEnd: [0, 0]
    // Avoid accuracy regression caused by this padding. Need specific handle:
    // MaxPool: If the input is all negative value the maximum value should also be negative
    //          If with an extra pad, the maximum value of the last row and column will be changed to 0
    // - The solution is to slice and concat input tensor instead of padding 0
    // AvgPool: The average weight for each value should be 1 / (KX * KY)
    //          If with an extra pad, the weight of last row and column will change to 1 / ((KX + padX) * (KY + padY))
    // - The solution is to use GroupConv with specific weight value instead of AvgPool
    const auto inShape = getShape(origOp.input());
    auto hasPadValue = llvm::any_of(kernelsInfoVal.padEnd, [](const int64_t padVal) {
        return padVal > 0;
    });

    if (hasPadValue) {
        // Only support global pooling with extra padding
        // For example: Pooling Op with kernel [23, 23], pads_begin = [0, 0], pads_end = [0, 0], strides = [23, 23]
        // Input shape: 1x16x46x46, Output shape: 1x16x2x2
        // TODO(E#90030): concat four global pooling with inshape 1x16x23x23 and strides [23, 23]
        if ((kernelsInfoVal.padEnd[Dims4D::PadsEnd::Bottom] > 0 &&
             inShape[Dims4D::Act::H] != origKernel[Dims4D::Kernel::Y.ind()]) ||
            (kernelsInfoVal.padEnd[Dims4D::PadsEnd::Right] > 0 &&
             inShape[Dims4D::Act::W] != origKernel[Dims4D::Kernel::X.ind()])) {
            rewriter.eraseOp(firstOpOutput.getDefiningOp());
            _log.trace("Cannot handle padding of '{1}' layer at '{2}'", origOp.getOperationName(), origOp->getLoc());
            return mlir::failure();
        }

        auto firstOp = mlir::dyn_cast<ConcreteOp>(firstOpOutput.getDefiningOp());
        VPUX_THROW_UNLESS(firstOp != nullptr, "Cannot get the first split op");
        firstOpOutput = handlePoolWithPadding(firstOp, rewriter);
    }

    // Step 3: Handle the second split Op
    auto secondOpOutput = createSecondSplitOp(origOp, firstOpOutput, kernelsInfoVal, rewriter);

    rewriter.replaceOp(origOp, secondOpOutput);
    return mlir::success();
}

//
// GeneralAvgPoolRewriter
//

class GeneralAvgPoolRewriter final : public GeneralPoolingBaseRewriter<IE::AvgPoolOp> {
public:
    GeneralAvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): GeneralPoolingBaseRewriter<IE::AvgPoolOp>(ctx, log) {
        setDebugName("AvgPoolRewriter");
    }

    mlir::Value handlePoolWithPadding(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const override;
};

mlir::Value GeneralAvgPoolRewriter::handlePoolWithPadding(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();

    const auto inShapeType = origOp.input().getType().cast<NDTypeInterface>();
    const auto inShape = inShapeType.getShape();
    const auto outShape = origOp.output().getType().cast<NDTypeInterface>().getShape();
    const auto weightsScaleVal = static_cast<float>(outShape[Dims4D::Act::H] * outShape[Dims4D::Act::W]) /
                                 static_cast<float>(inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W]);

    const auto kernel = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto weightShape = SmallVector<int64_t>{inShape[Dims4D::Act::C], 1, kernel[Dims4D::Kernel::Y.ind()],
                                                  kernel[Dims4D::Kernel::X.ind()]};
    const auto dataStorageType = mlir::RankedTensorType::get(weightShape, inShapeType.getElementType());
    const auto denseElementVal = mlir::DenseElementsAttr::get(dataStorageType, static_cast<float16>(weightsScaleVal));
    auto weights = rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataStorageType,
                                                     Const::ContentAttr::get(denseElementVal));

    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto groupAttr = getIntAttr(ctx, inShape[Dims4D::Act::C]);

    _log.trace("Replace AvgPoolOp has Padding value with GroupConv to avoid accuracy issue");
    return rewriter
            .replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, origOp.input(), weights,
                                                        /*bias=*/nullptr, origOp.stridesAttr(), origOp.pads_beginAttr(),
                                                        origOp.pads_endAttr(), dilationsAttr, groupAttr,
                                                        /*post_opAttr=*/nullptr)
            .output();
}

//
// GeneralMaxPoolRewriter
//

class GeneralMaxPoolRewriter final : public GeneralPoolingBaseRewriter<IE::MaxPoolOp> {
public:
    GeneralMaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): GeneralPoolingBaseRewriter<IE::MaxPoolOp>(ctx, log) {
        setDebugName("GeneralMaxPoolRewriter");
    }

    mlir::Value handlePoolWithPadding(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const override;
};

mlir::Value GeneralMaxPoolRewriter::handlePoolWithPadding(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();

    const auto inShape = origOp.input().getType().cast<NDTypeInterface>().getShape();
    auto paddingActivation = [&](mlir::Value input, const int64_t padSize, const Dim padDim) -> mlir::Value {
        auto offsets = SmallVector<int64_t>(inShape.size(), 0);
        auto sliceShape = to_small_vector(inShape.raw());
        sliceShape[padDim.ind()] = padSize;
        auto sliceOp = rewriter.create<IE::SliceOp>(origOp.getLoc(), input, getIntArrayAttr(ctx, offsets),
                                                    getIntArrayAttr(ctx, sliceShape));

        return rewriter.create<IE::ConcatOp>(origOp.getLoc(), SmallVector<mlir::Value>{input, sliceOp.result()}, padDim)
                .output();
    };

    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    auto inputVal = origOp.input();
    if (padEnd[Dims4D::PadsEnd::Right.ind()] > 0) {
        inputVal = paddingActivation(inputVal, padEnd[Dims4D::PadsEnd::Right.ind()], Dims4D::Act::W);
    }

    if (padEnd[Dims4D::PadsEnd::Bottom.ind()] > 0) {
        inputVal = paddingActivation(inputVal, padEnd[Dims4D::PadsEnd::Bottom.ind()], Dims4D::Act::H);
    }

    _log.trace("Replace MaxPoolOp has Padding value with Slice and Concat to avoid accuracy issue");
    return rewriter
            .replaceOpWithNewOp<IE::MaxPoolOp>(origOp, inputVal, origOp.kernel_sizeAttr(), origOp.stridesAttr(),
                                               origOp.pads_beginAttr(), getIntArrayAttr(ctx, makeArrayRef({0, 0})),
                                               origOp.rounding_type(), origOp.post_opAttr())
            .output();
}

//
// OverlappedMaxPoolRewriter
//

class OverlappedMaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    OverlappedMaxPoolRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("OverlappedMaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOverlappedMaxPool(IE::MaxPoolOp origOp) const;

private:
    Logger _log;
};

// Currently only support specific MaxPool with overlapped data
// For example: MaxPool Op with kernel [13, 13], pads_begin = [6, 6], pads_end = [6, 6], strides = [1, 1]
// - Split 1: Kernel: [7, 7], Stride: [1, 1], PadBegin: [3, 3], PadEnd: [3, 3]
// - Split 2: Kernel: [7, 7], Stride: [1, 1], PadBegin: [3, 3], PadEnd: [3, 3]
// If there are more related cases in the future. This part can be considered a more general implementation.
bool OverlappedMaxPoolRewriter::isLegalOverlappedMaxPool(IE::MaxPoolOp origOp) const {
    const auto inShape = origOp.input().getType().cast<vpux::NDTypeInterface>().getShape();
    if (inShape.size() != 4) {
        _log.trace("Overlapped MaxPool Op should with input shape rank equal 4");
        return true;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    if (vpux::IE::hasSupportedKernels(Shape(kernelSize))) {
        _log.trace("Kernel size of Overlapped MaxPool Op is legal");
        return true;
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto strideSize = parseIntArrayAttr<int64_t>(origOp.strides());

    const auto isLargerThanOne = [](auto stride) {
        return stride > 1;
    };

    if (llvm::any_of(strideSize, isLargerThanOne)) {
        _log.trace("Overlapped MaxPool should with strides equal 1");
        return true;
    }

    const auto isEvenNumb = [](const int64_t val) {
        return val % 2 == 0;
    };

    if (llvm::any_of(kernelSize, isEvenNumb) || !llvm::all_of(padBegin, isEvenNumb)) {
        _log.trace("Overlapped MaxPool should with Kernel is even number and Pad is odd number");
        return true;
    }

    auto isSupportedRelation = [](int64_t kernelVal, int64_t padVal) {
        return kernelVal == padVal * 2 + 1;
    };

    auto isSupportedOverlappedMaxPool =
            isSupportedRelation(kernelSize[Dims4D::Kernel::X.ind()], padBegin[Dims4D::PadsBegin::Left.ind()]) &&
            isSupportedRelation(kernelSize[Dims4D::Kernel::X.ind()], padBegin[Dims4D::PadsEnd::Right.ind()]) &&
            isSupportedRelation(kernelSize[Dims4D::Kernel::Y.ind()], padBegin[Dims4D::PadsBegin::Top.ind()]) &&
            isSupportedRelation(kernelSize[Dims4D::Kernel::Y.ind()], padBegin[Dims4D::PadsEnd::Bottom.ind()]);

    if (!isSupportedOverlappedMaxPool) {
        _log.trace("Overlapped MaxPool should with Kernel size and Pad size: kernelVal -1 = 2 * padVal");
        return true;
    }

    return false;
}

mlir::LogicalResult OverlappedMaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' layer at '{2}'", getDebugName(), origOp.getOperationName(), origOp->getLoc());

    auto* ctx = origOp->getContext();

    if (isLegalOverlappedMaxPool(origOp)) {
        return mlir::failure();
    }

    const auto kernelsSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto padsBeginVal = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEndVal = parseIntArrayAttr<int64_t>(origOp.pads_end());

    const auto newKernelsAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{(kernelsSize[Dims4D::Kernel::Y.ind()] + 1) / 2,
                                                      (kernelsSize[Dims4D::Kernel::X.ind()] + 1) / 2});
    const auto newPadsBeginAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{padsBeginVal[Dims4D::PadsBegin::Top.ind()] / 2,
                                                      padsBeginVal[Dims4D::PadsBegin::Left.ind()] / 2});
    const auto newPadsEndAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{padsEndVal[Dims4D::PadsEnd::Bottom.ind()] / 2,
                                                      padsEndVal[Dims4D::PadsEnd::Right.ind()] / 2});

    auto firstOp = rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), origOp.input(), newKernelsAttr, origOp.stridesAttr(),
                                                  newPadsBeginAttr, newPadsEndAttr, origOp.rounding_type(),
                                                  origOp.post_opAttr());
    _log.nest(2).trace("Create firstSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
                       newKernelsAttr, origOp.stridesAttr(), newPadsBeginAttr, newPadsEndAttr);

    _log.nest(2).trace("Create secondSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
                       newKernelsAttr, origOp.stridesAttr(), newPadsBeginAttr, newPadsEndAttr);
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(origOp, firstOp.output(), newKernelsAttr, origOp.stridesAttr(),
                                               newPadsBeginAttr, newPadsEndAttr, origOp.rounding_type(),
                                               origOp.post_opAttr());

    return mlir::success();
}

//
// ConvRewriter
//

class ConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

    SmallVector<mlir::Value> sliceFilter(const mlir::Value filterToSplit, const int64_t numXSlices,
                                         const int64_t numYSlices, const int64_t targetKernelSize,
                                         const mlir::Location location, mlir::PatternRewriter& rewriter) const;

    mlir::Value getExtendedActivation(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const;

    void rewriteSubGraph(IE::ConvolutionOp origOp, ArrayRef<mlir::Value> slicedFilters, mlir::Value newActivation,
                         int64_t numXSlices, const int64_t numYSlices, const int64_t targetKernelSize,
                         mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::Value getZerosConst(mlir::PatternRewriter& rewriter, ShapeRef constShape, IE::ConvolutionOp origOp) {
    const auto elemType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(constShape), elemType);

    mlir::DenseElementsAttr denseElementVal = wrapData(dataStorageType, 0.0f);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "ConvolutionOp has incompatible data type {0}, only float16 or float32 are supported", elemType);

    return rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, Const::ContentAttr::get(denseElementVal))
            .getOutput();
}

SmallVector<mlir::Value> ConvRewriter::sliceFilter(const mlir::Value filterToSplit, const int64_t numXSlices,
                                                   const int64_t numYSlices, const int64_t targetKernelSize,
                                                   const mlir::Location location,
                                                   mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    SmallVector<mlir::Value> slicedFilters;

    const auto filterShape = getShape(filterToSplit);
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto KY = filterShape[Dims4D::Filter::KY];

    for (int64_t j = 0; j < numYSlices; j++) {
        for (int64_t i = 0; i < numXSlices; i++) {
            int64_t slicedKX = std::min(KX, targetKernelSize);
            if (i == (numXSlices - 1)) {
                slicedKX = KX - (numXSlices - 1) * targetKernelSize;
            }

            int64_t slicedKY = std::min(KY, targetKernelSize);
            if (j == (numYSlices - 1)) {
                slicedKY = KY - (numYSlices - 1) * targetKernelSize;
            }

            const auto IC = filterShape[Dims4D::Filter::IC];
            const auto OC = filterShape[Dims4D::Filter::OC];
            SmallVector<int64_t> sliceShape{OC, IC, slicedKY, slicedKX};

            Shape offsets(filterShape.size());
            offsets[Dims4D::Filter::KX] = i * targetKernelSize;
            offsets[Dims4D::Filter::KY] = j * targetKernelSize;
            auto slice = rewriter.create<IE::SliceOp>(location, filterToSplit, getIntArrayAttr(ctx, offsets.raw()),
                                                      getIntArrayAttr(ctx, sliceShape));
            slicedFilters.push_back(slice);
        }
    }
    return slicedFilters;
}

mlir::Value ConvRewriter::getExtendedActivation(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> extendedActivation;

    auto activation = origOp->getOperand(0);
    const auto inputShape = getShape(activation);
    int64_t newWidth = inputShape[Dims4D::Act::W];

    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.pads_begin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.pads_end()));

    auto const extendActivationOnWidth = [&](int64_t dim) {
        Shape zeroConstShape(to_small_vector(inputShape));
        zeroConstShape[Dims4D::Act::W] = dim;
        auto constZeros = getZerosConst(rewriter, zeroConstShape, origOp);
        extendedActivation.push_back(constZeros);
        newWidth += dim;
    };

    if (padStart[Dims4D::PadsBegin::Left] > 0) {
        extendActivationOnWidth(padStart[Dims4D::PadsBegin::Left]);
    }

    extendedActivation.push_back(activation);

    if (padEnd[Dims4D::PadsEnd::Right] > 0) {
        extendActivationOnWidth(padStart[Dims4D::PadsEnd::Right]);
    }

    auto tempActivation =
            rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), mlir::ValueRange(extendedActivation), Dims4D::Act::W);

    extendedActivation.clear();

    auto const extendActivationOnHeight = [&](int64_t dim) {
        Shape zeroConstShape(to_small_vector(inputShape));
        zeroConstShape[Dims4D::Act::H] = dim;
        zeroConstShape[Dims4D::Act::W] = newWidth;
        auto constZeros = getZerosConst(rewriter, zeroConstShape, origOp);
        extendedActivation.push_back(constZeros);
    };
    if (padStart[Dims4D::PadsBegin::Top] > 0) {
        extendActivationOnHeight(padStart[Dims4D::PadsBegin::Top]);
    }

    extendedActivation.push_back(tempActivation);

    if (padEnd[Dims4D::PadsEnd::Bottom] > 0) {
        extendActivationOnHeight(padStart[Dims4D::PadsEnd::Bottom]);
    }

    return rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), mlir::ValueRange(extendedActivation), Dims4D::Act::H);
}

void ConvRewriter::rewriteSubGraph(IE::ConvolutionOp origOp, ArrayRef<mlir::Value> slicedFilters,
                                   mlir::Value extendedActivation, int64_t numXSlices, const int64_t numYSlices,
                                   const int64_t targetKernelSize, mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    const auto inputShape = getShape(origOp->getOperand(0));
    const auto filterShape = getShape(origOp.filter());
    const auto origKX = filterShape[Dims4D::Filter::KX];
    const auto origKY = filterShape[Dims4D::Filter::KY];
    const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
    const auto broadcastType =
            vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    const auto extendedActivationShape = getShape(extendedActivation);

    SmallVector<mlir::Value> accumulativeOutputTensors;
    for (int64_t j = 0; j < numYSlices; j++) {
        for (int64_t i = 0; i < numXSlices; i++) {
            int64_t startW = i * targetKernelSize;
            VPUX_THROW_WHEN(startW >= extendedActivationShape[Dims4D::Act::W], "dimension W out of range");
            int64_t startH = j * targetKernelSize;
            VPUX_THROW_WHEN(startH >= extendedActivationShape[Dims4D::Act::H], "dimension H out of range");

            auto slicedFilterShape = getShape(slicedFilters[j * numXSlices + i]);
            // Calculate activation slice shape
            int64_t newActivationWidth =
                    ((extendedActivationShape[Dims4D::Act::W] - origKX) / strides[Dims4D::Strides::X]) *
                            strides[Dims4D::Strides::X] +
                    slicedFilterShape[Dims4D::Act::W];
            int64_t newActivationHeight =
                    ((extendedActivationShape[Dims4D::Act::H] - origKY) / strides[Dims4D::Strides::Y]) *
                            strides[Dims4D::Strides::Y] +
                    slicedFilterShape[Dims4D::Act::H];
            if (newActivationWidth > extendedActivationShape[Dims4D::Act::W]) {
                newActivationWidth = extendedActivationShape[Dims4D::Act::W];
            }
            if (newActivationHeight > extendedActivationShape[Dims4D::Act::H]) {
                newActivationHeight = extendedActivationShape[Dims4D::Act::H];
            }

            mlir::Value convInput;
            SmallVector<int64_t> sliceShape{extendedActivationShape[Dims4D::Act::N],
                                            extendedActivationShape[Dims4D::Act::C], newActivationHeight,
                                            newActivationWidth};
            Shape offsets(inputShape.size());
            offsets[Dims4D::Act::W] = startW;
            offsets[Dims4D::Act::H] = startH;
            _log.trace("[{0}] Activation slice shape {1}, slice offsets {2}", getDebugName(), sliceShape, offsets);

            convInput =
                    rewriter.create<IE::SliceOp>(origOp->getLoc(), extendedActivation,
                                                 getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            // Add bias and post process for the last convolution and eltwise.
            auto isLastSlice = i == (numXSlices - 1) && j == (numYSlices - 1);
            auto conv = rewriter.create<IE::ConvolutionOp>(
                    origOp->getLoc(), convInput, slicedFilters[j * numXSlices + i],
                    isLastSlice ? origOp.bias() : nullptr, origOp.strides(),
                    getIntArrayAttr(origOp->getContext(), makeArrayRef({0, 0})),
                    getIntArrayAttr(origOp->getContext(), makeArrayRef({0, 0})), origOp.dilationsAttr(), nullptr);

            if (!accumulativeOutputTensors.empty()) {
                auto add = rewriter.create<IE::AddOp>(origOp->getLoc(), accumulativeOutputTensors.back(), conv,
                                                      broadcastType, isLastSlice ? origOp.post_opAttr() : nullptr);
                accumulativeOutputTensors.push_back(add);
            } else {
                accumulativeOutputTensors.push_back(conv);
            }
        }
    }

    _log.trace("[{0}] Successufuly replace large convolution at {1}", getDebugName(), origOp->getLoc());

    rewriter.replaceOp(origOp, accumulativeOutputTensors.back());
}

mlir::LogicalResult ConvRewriter::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());
    const auto filterShape = getShape(origOp.filter());
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto KY = filterShape[Dims4D::Filter::KY];

    auto targetKernelSize = VPU::NCEInvariant::MAX_KERNEL_SIZE;
    auto numXSlices = checked_cast<int64_t>(llvm::alignTo(KX, targetKernelSize) / targetKernelSize);
    auto numYSlices = checked_cast<int64_t>(llvm::alignTo(KY, targetKernelSize) / targetKernelSize);

    // Slice filter
    SmallVector<mlir::Value> slicedFilters =
            sliceFilter(origOp.filter(), numXSlices, numYSlices, targetKernelSize, origOp->getLoc(), rewriter);
    _log.trace("[{0}] Split kernel into {1} small kernels {2} at {3}", getDebugName(), slicedFilters.size(),
               slicedFilters, origOp->getLoc());

    // Pad activation
    auto extendedActivation = getExtendedActivation(origOp, rewriter);
    _log.trace("[{0}] Pad on activation, new shape {1}, new activation {2} at {3}", getDebugName(),
               getShape(extendedActivation), extendedActivation, origOp->getLoc());

    // Create new sub graph and replace origOp
    rewriteSubGraph(origOp, slicedFilters, extendedActivation, numXSlices, numYSlices, targetKernelSize, rewriter);
    return mlir::success();
}

//
// HandleLargeKernelsPass
//

class HandleLargeKernelsPass final : public IE::HandleLargeKernelsBase<HandleLargeKernelsPass> {
public:
    explicit HandleLargeKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleLargeKernelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<Const::DeclareOp>();

    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        auto activationRank = op.input().getType().cast<vpux::NDTypeInterface>().getRank();
        auto filterRank = op.filter().getType().cast<vpux::NDTypeInterface>().getRank();
        if (activationRank != 4 || filterRank != 4) {
            return true;
        }

        const auto filterShape = getShape(op.filter());
        const auto KY = filterShape[Dims4D::Filter::KY];
        const auto KX = filterShape[Dims4D::Filter::KX];

        return KY <= VPU::NCEInvariant::MAX_KERNEL_SIZE && KX <= VPU::NCEInvariant::MAX_KERNEL_SIZE;
    });

    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    // Note 1: Each execution of matchAndRewrite will only split Pooling Op into two
    // In some cases, it need continue splitting until all the Op are legal
    // So used GreedyRewriteConfig here
    // Note 2: The reason to distinguish GeneralPool and OverlappedPool is the influence of padding Attr.
    // The pad will have different implementation methods. To simplify and split it.
    mlir::RewritePatternSet poolingPatterns(&ctx);
    poolingPatterns.add<GeneralAvgPoolRewriter>(&ctx, _log);
    poolingPatterns.add<GeneralMaxPoolRewriter>(&ctx, _log);
    poolingPatterns.add<OverlappedMaxPoolRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(poolingPatterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleLargeKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleLargeKernelsPass(Logger log) {
    return std::make_unique<HandleLargeKernelsPass>(log);
}
