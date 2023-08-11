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

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

constexpr int64_t PADDING_RIGHT = 1;
constexpr int64_t PADDING_BOT = 3;

//
// Assign parameter `stride = 1` rather than firstOpKernel or sequencedOpKernel if axis of sequenced pools is global.
//
mlir::ArrayAttr getGlobalOrOrigStride(mlir::MLIRContext* ctx, mlir::Value input, std::array<int64_t, 2> origKernel,
                                      std::array<int64_t, 2> origStride, mlir::ArrayAttr padBeginAttr,
                                      mlir::ArrayAttr padEndAttr) {
    std::array<int64_t, 2> newOpStride = {1, 1};
    auto origInputShape = getShape(input).raw();

    auto padBegin = parseIntArrayAttr<int64_t>(padBeginAttr);
    auto padEnd = parseIntArrayAttr<int64_t>(padEndAttr);

    if (origKernel[Dims4D::Kernel::Y.ind()] !=
        (origInputShape[Dims4D::Act::H.ind()] + padBegin[Dims4D::PadsBegin::Top.ind()] +
         padEnd[Dims4D::PadsEnd::Bottom.ind()]))
        newOpStride[Dims4D::Strides::Y.ind()] = origStride[Dims4D::Strides::Y.ind()];

    if (origKernel[Dims4D::Kernel::X.ind()] !=
        (origInputShape[Dims4D::Act::W.ind()] + padBegin[Dims4D::PadsBegin::Left.ind()] +
         padEnd[Dims4D::PadsEnd::Right.ind()]))
        newOpStride[Dims4D::Strides::X.ind()] = origStride[Dims4D::Strides::X.ind()];

    return getIntArrayAttr(ctx, makeArrayRef(newOpStride));
}

void getFactorsForSecondDimension(std::array<int64_t, 4>& padding, std::array<int64_t, 2>& firstOpKernel,
                                  std::array<int64_t, 2>& sequencedOpKernel, int32_t smallDim, Logger log,
                                  ArrayRef<int64_t> kernelSize) {
    int64_t padValue = 1;
    const auto factorsResult =
            vpux::IE::getFactors(kernelSize[smallDim], padValue);  // toggling between the two kernel sizes

    VPUX_THROW_UNLESS(
            factorsResult.hasValue(),
            "Failed to get valid factors when splitting kernel! Large padding value would lead to accuracy drop.");

    const auto factorsSecondDim = factorsResult.getValue();

    log.trace("Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}", smallDim,
              kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);

    VPUX_THROW_UNLESS((factorsSecondDim.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE) &&
                              (factorsSecondDim.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE),
                      "Second dimension factors ({1}, {2})  are larger than MAX_KERNEL_SIZE {0}",
                      VPU::NCEInvariant::MAX_KERNEL_SIZE, factorsSecondDim.larger, factorsSecondDim.smaller);
    firstOpKernel[smallDim] = factorsSecondDim.larger;
    sequencedOpKernel[smallDim] = factorsSecondDim.smaller;
    auto multipliedFactors = firstOpKernel[smallDim] * sequencedOpKernel[smallDim];

    padding[PADDING_BOT] = (multipliedFactors > kernelSize[smallDim]) ? 1 : 0;
}

void getFactorsForSecondDimensionWithLimit(std::array<int64_t, 4>& padding, std::array<int64_t, 2>& firstOpKernel,
                                           std::array<int64_t, 2>& sequencedOpKernel, int32_t smallDim, Logger log,
                                           ArrayRef<int64_t> kernelSize) {
    int64_t padValue = 1;
    auto factorsResult = vpux::IE::getFactors(kernelSize[smallDim], padValue);  // toggling between the two kernel sizes
    if (!factorsResult.hasValue()) {
        padValue = 1;
        factorsResult = vpux::IE::getFactorsWithSupportedLarger(kernelSize[smallDim], padValue);
    }

    VPUX_THROW_UNLESS(
            factorsResult.hasValue(),
            "Failed to get valid factors when splitting kernel! Large padding value would lead to accuracy drop.");

    const auto factorsSecondDim = factorsResult.getValue();

    if (factorsSecondDim.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE) {
        log.trace("Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}", smallDim,
                  kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);
    } else {
        log.trace(
                "Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}(Required further split!)",
                smallDim, kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);
    }

    VPUX_THROW_UNLESS(factorsSecondDim.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE,
                      "Second dimension factors ({1}, {2})  are larger than MAX_KERNEL_SIZE {0}",
                      VPU::NCEInvariant::MAX_KERNEL_SIZE, factorsSecondDim.larger, factorsSecondDim.smaller);
    firstOpKernel[smallDim] = factorsSecondDim.larger;
    sequencedOpKernel[smallDim] = factorsSecondDim.smaller;
    auto multipliedFactors = firstOpKernel[smallDim] * sequencedOpKernel[smallDim];

    padding[PADDING_BOT] = (multipliedFactors > kernelSize[smallDim]) ? 1 : 0;
}

bool isLegalAvgPoolOp(IE::AvgPoolOp op, Logger log) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
    if (vpux::IE::hasSupportedKernels(kernelSize)) {
        return true;
    }
    const auto inDataType = op.input().getType().cast<vpux::NDTypeInterface>();
    const auto inDataShape = inDataType.getShape().raw();
    const auto strides = parseIntArrayAttr<int64_t>(op.strides());
    const auto inputElemType = inDataType.getElementType();
    auto unsupportedKernelCheck = [&](int32_t kernelInd, int32_t actInd, int32_t strideInd) {
        // Support multiple splitting for larger kernel size (> 11 * 11) with FP16/FP32 input as no drop in accuracy
        if (inputElemType.isF16() || inputElemType.isF32()) {
            return (kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]);
        } else {
            const auto maxKernelSizeSupported =
                    VPU::NCEInvariant::MAX_KERNEL_SIZE *
                    VPU::NCEInvariant::MAX_KERNEL_SIZE;  // we can only get 2 factors
                                                         // and max kernel should be 11 * 11 = 121
            return ((kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]) ||
                    kernelSize[kernelInd] > maxKernelSizeSupported);
        }
    };

    if (unsupportedKernelCheck(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind())) {
        log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
        return true;
    }
    if (unsupportedKernelCheck(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
        log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
        return true;
    }
    // In these cases, more performant to execute this AvgPool on shave
    // leave it on for VPUX3700 as soon as VPUX3720 have HW AVG
    const auto arch = VPU::getArch(op);
    if (arch != VPU::ArchKind::VPUX37XX &&
        (kernelSize[Dims4D::Kernel::X.ind()] == 1 || kernelSize[Dims4D::Kernel::Y.ind()] == 1)) {
        log.trace("AvgPool operation ignored by HandleLargeKernel pass for performance");
        return true;
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
    const auto zeros = SmallVector<int64_t>{0, 0};
    if ((padsBegin != zeros || padsEnd != zeros) && op.exclude_pads()) {
        return true;
    }
    return false;
}

void calculateKernelsAndPadding(ArrayRef<int64_t> kernelSize, std::array<int64_t, 4>& padding,
                                std::array<int64_t, 2>& firstOpKernel, std::array<int64_t, 2>& sequencedOpKernel,
                                bool supportMultipleSplitting, Logger log) {
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

    // figure out the bigger kernel dimension width or height when having an asymmetric kernel
    auto largerKernelSize = KX;
    auto largeDim = Dims4D::Kernel::X.ind();
    auto smallDim = Dims4D::Kernel::Y.ind();
    auto asymmetricCase = (KX != KY);
    auto asymmetricBothKernelsLarge =
            (asymmetricCase && (KX > VPU::NCEInvariant::MAX_KERNEL_SIZE) && (KY > VPU::NCEInvariant::MAX_KERNEL_SIZE));

    // deal with asymmetric kernels when one dim is larger than MAX_KERNEL_SIZE
    if (asymmetricCase && (KX < KY)) {
        largerKernelSize = KY;
        largeDim = Dims4D::Kernel::Y.ind();
        smallDim = Dims4D::Kernel::X.ind();
    }
    int64_t padValue = 1;
    auto factorsResult = vpux::IE::getFactors(largerKernelSize, padValue);
    if (!factorsResult.hasValue() && supportMultipleSplitting) {
        padValue = 1;
        factorsResult = vpux::IE::getFactorsWithSupportedLarger(largerKernelSize, padValue);
    }
    VPUX_THROW_UNLESS(
            factorsResult.hasValue(),
            "Failed to get valid factors when splitting kernel! Large padding value would lead to accuracy drop.");

    const auto factors = factorsResult.getValue();

    if (factors.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE) {
        log.trace("Large Dimension kernelSize[{0}] = {1}, larger factor: {2} , smaller factor: {3}", largeDim,
                  largerKernelSize, factors.larger, factors.smaller);
    } else {
        log.trace("Large Dimension kernelSize[{0}] = {1}, larger factor: {2} , smaller factor: {3}"
                  "(Required further split!)",
                  largeDim, largerKernelSize, factors.larger, factors.smaller);
    }
    VPUX_THROW_UNLESS(factors.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE,
                      "Large dimension factors ({1}, {2})  are larger the MAX_KERNEL_SIZE {0}",
                      VPU::NCEInvariant::MAX_KERNEL_SIZE, factors.larger, factors.smaller);

    // cascading supported ops
    // first op kernel [factors.larger, factorsSecondDim.larger] - firstOpKernel
    // sequenced op kernel [factors.smaller, factorsSecondDim.smaller] - sequencedOpKernel
    // Padding quantity relationship is (input size + pad) / k = output size, padding config is TRUE, FALSE
    firstOpKernel[largeDim] = factors.larger;  // first was the large dimension
    sequencedOpKernel[largeDim] = factors.smaller;
    auto multipliedFactors = firstOpKernel[largeDim] * sequencedOpKernel[largeDim];

    if (asymmetricCase) {
        if (asymmetricBothKernelsLarge) {
            if (factors.smaller > VPU::NCEInvariant::MAX_KERNEL_SIZE) {
                getFactorsForSecondDimensionWithLimit(padding, firstOpKernel, sequencedOpKernel, smallDim, log,
                                                      kernelSize);
            } else {
                getFactorsForSecondDimension(padding, firstOpKernel, sequencedOpKernel, smallDim, log, kernelSize);
            }
        } else {
            firstOpKernel[smallDim] = kernelSize[smallDim];
            sequencedOpKernel[smallDim] =
                    1;  // the smallDim was not factorized, the multiplication kSize*1 covers the second op

            padding[PADDING_BOT] = 0;
        }
        // factors multiplied > kernel, we need padding
        padding[PADDING_RIGHT] = (multipliedFactors > kernelSize[largeDim]) ? padValue : 0;

        if (largeDim != Dims4D::Kernel::X.ind()) {
            // change the padding on the other dimensions as largeDim was not on the width dimension - PADD_RIGHT
            std::swap(padding[PADDING_RIGHT], padding[PADDING_BOT]);
        }
    } else {
        firstOpKernel[smallDim] = factors.larger;  // largeDim has the same kernel size as smallDim
        sequencedOpKernel[smallDim] = factors.smaller;
        padding[PADDING_RIGHT] = padding[PADDING_BOT] = (multipliedFactors > kernelSize[largeDim]) ? padValue : 0;
    }
}

//
// AveragePoolRewriter
//

class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

    mlir::FailureOr<mlir::Value> splitAvgOperationSlicing(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AveragePool layer at '{1}'", getDebugName(), origOp->getLoc());

    if (isLegalAvgPoolOp(origOp, _log)) {
        return mlir::failure();
    }
    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel = {1, 1}, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    auto* ctx = origOp->getContext();

    auto curOpInput = origOp.input();
    auto curKernelSize = kernelSize;

    // Support multiple splitting for larger kernel size (> 11 * 11)
    // For example, kernel = 128, it will return [8, 16] factors in first round splitting
    // The supported factor 8: it will be used for current AvgPool kernel
    // The unsupported factor 16 (>11): it will be splitted to [4, 4] in the next round splitting
    // FP16/FP32 input: multiple splitting will introduce accuracy loss as Min/Max changed for FP16-INT8 model
    bool supportMultipleSplitting = false;
    // If a pad is needed, padding 0 will in averaging calculation. For averaging calculation, in this case, the
    // numerator remains the same, the denominator becomes larger. The accuracy of the results is not correct. So use
    // convolution instead.
    auto convertToConv = false;

    const auto inDataType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputElemType = inDataType.getElementType();
    if (inputElemType.isF16() || inputElemType.isF32()) {
        supportMultipleSplitting = true;
    }

    calculateKernelsAndPadding(curKernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel,
                               supportMultipleSplitting, _log.nest(2));
    const auto KY = curKernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = curKernelSize[Dims4D::Kernel::X.ind()];
    const auto inShape = getShape(curOpInput);

    // The first kernel has stride size same as kernel size. For a global AveragePool with symmetric kernel,
    // if the first kernel is larger than MAX_STRIDE, it can't be converted to NCE task. However, we can
    // reverse kernel order to avoid large stride on the first kernel. For example, when we split a large
    // kernel of 176x176, the kernel size would be 11x11, 4x4, 4x4 sequence with [11, 11], [4, 4], [1, 1]
    // stride size sequence. The first kernel has a large stride so it can't be converted to NCE task. However,
    // if we reverse kernel order. When the first stage splitting, 176x176 would be splitted to 11x11, 16x16
    // with stride size [11, 11], [16, 16] (16x16 will be splitted in the second stage). Stride size [11, 11]
    // is not supported by NCE so we reverse the sequence to 16x16, 11x11 with stride size [16, 16], [1, 1].
    // And after the second stage splitting, it will be splitted into 4x4, 4x4, 11x11 with stride size
    // [4, 4], [4,4], [1, 1]. Both of them are supported by NCE.
    if ((KY == KX) && (inShape[Dims4D::Act::H] == KY && inShape[Dims4D::Act::W] == KX)) {
        if (firstOpKernel[Dims4D::Kernel::Y.ind()] > VPU::NCEInvariant::MAX_STRIDE &&
            firstOpKernel[Dims4D::Kernel::Y.ind()] <= VPU::NCEInvariant::MAX_KERNEL_SIZE) {
            std::swap(firstOpKernel, sequencedOpKernel);
        }
        for (auto pad : calculatedPadding) {
            if (pad != 0) {
                convertToConv = true;
            }
        }
    }

    rewriter.setInsertionPoint(origOp);
    const auto firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
    const auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));
    const auto firstOpStrideAttr =
            getGlobalOrOrigStride(ctx, curOpInput, firstOpKernel, firstOpKernel, firstOpPadBegin, firstOpPadEnd);

    mlir::Value firstOpOutput;

    if (convertToConv) {
        const auto groupAttr = getIntAttr(ctx, inShape[Dims4D::Act::C]);
        auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int32_t>{1, 1});

        const ngraph::float16 weightsScaleVal = static_cast<float>(sequencedOpKernel[0] * sequencedOpKernel[1]) /
                                                static_cast<float>(inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W]);

        const auto elemType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
        const SmallVector<int64_t> weightShape = {inShape[Dims4D::Act::C], 1, firstOpKernel[0], firstOpKernel[1]};

        const auto dataStorageType = mlir::RankedTensorType::get(weightShape, elemType);
        const auto denseElementVal = mlir::DenseElementsAttr::get(dataStorageType, weightsScaleVal);
        auto firstOpWeights = rewriter.create<Const::DeclareOp>(origOp->getLoc(), dataStorageType,
                                                                Const::ContentAttr::get(denseElementVal))
                                      .output();
        auto firstGroupConvOp = rewriter.create<IE::GroupConvolutionOp>(
                origOp->getLoc(), curOpInput, firstOpWeights,
                /*bias=*/nullptr, firstOpStrideAttr, firstOpPadBegin, firstOpPadEnd, dilationsAttr, groupAttr,
                /*post_opAttr=*/nullptr);
        firstOpOutput = firstGroupConvOp.output();
        _log.trace("create firstGroupConvOp '{0}'", firstGroupConvOp);
    } else {
        const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));

        auto firstAvgPoolOp = rewriter.create<IE::AvgPoolOp>(
                origOp->getLoc(), curOpInput, firstOpKernelAttr, firstOpStrideAttr, firstOpPadBegin, firstOpPadEnd,
                origOp.rounding_typeAttr(), origOp.exclude_padsAttr(), origOp.post_opAttr());
        firstOpOutput = firstAvgPoolOp.output();
        _log.trace("create firstAvgPoolOp '{0}'", firstAvgPoolOp);

        // VPUX3720 are support different XY strides
        const auto arch = VPU::getArch(origOp);
        if (arch != VPU::ArchKind::VPUX37XX) {
            auto checkStrideRelation = [](const int64_t strideLeft, const int64_t strideRight) -> bool {
                return strideLeft > strideRight && strideLeft % strideRight == 0;
            };

            bool useSplitAvgOperationSlicing = checkStrideRelation(firstOpKernel[Dims4D::Strides::X.ind()],
                                                                   firstOpKernel[Dims4D::Strides::Y.ind()]) ||
                                               checkStrideRelation(firstOpKernel[Dims4D::Strides::Y.ind()],
                                                                   firstOpKernel[Dims4D::Strides::X.ind()]);
            if (useSplitAvgOperationSlicing) {
                const auto concatOp = splitAvgOperationSlicing(firstAvgPoolOp, rewriter);
                if (mlir::failed(concatOp)) {
                    return mlir::failure();
                }
                firstOpOutput = concatOp.getValue();
            }
        }
    }

    const auto firstOpOutputShapeType = firstOpOutput.getType().cast<vpux::NDTypeInterface>();
    const auto firstOpOutputShape = firstOpOutputShapeType.getShape().raw();

    auto globalAvgOverH = firstOpOutputShape[Dims4D::Act::H.ind()] == sequencedOpKernel[0];
    auto globalAvgOverW = firstOpOutputShape[Dims4D::Act::W.ind()] == sequencedOpKernel[1];

    std::array<int64_t, 2> sequencedOpStrides = {1, 1};
    if (!globalAvgOverH) {
        sequencedOpStrides[0] = sequencedOpKernel[0];
    }
    if (!globalAvgOverW) {
        sequencedOpStrides[1] = sequencedOpKernel[1];
    }

    calculatedPadding = {0, 0, 0, 0};
    const auto sequencedOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto sequencedOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));

    const auto sequencedOpStridesAttr = getGlobalOrOrigStride(ctx, firstOpOutput, sequencedOpKernel, sequencedOpStrides,
                                                              sequencedOpPadBegin, sequencedOpPadEnd);
    auto sequenceOp = rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
            origOp, origOp.getType(), firstOpOutput, sequencedOpKernelAttr, sequencedOpStridesAttr, sequencedOpPadBegin,
            sequencedOpPadEnd, origOp.rounding_typeAttr(), origOp.exclude_padsAttr(), origOp.post_opAttr());
    _log.trace("create sequenceOp '{0}'", sequenceOp);

    return mlir::success();
}

mlir::FailureOr<mlir::Value> AveragePoolRewriter::splitAvgOperationSlicing(IE::AvgPoolOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    const auto inputShape = getShape(origOp.input());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    if (strides[0] <= 0 || strides[1] <= 0) {
        return errorAt(origOp->getLoc(), "Invalid stride value");
    }
    const auto minStride = std::min(strides[0], strides[1]);
    const auto maxStride = std::max(strides[0], strides[1]);
    auto paddingEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());

    // calculate the new stride for avg pooling
    const auto newStrides = getIntArrayAttr(ctx, makeArrayRef({maxStride, maxStride}));

    // try to slice the tensor into maxStride/minStride pieces on the dim with minStride, and don't need slice on the
    // other dim
    int64_t stepsH = (strides[1] + strides[0] - 1) / strides[0];  // the slice number on the height axis
    int64_t stepsW = (strides[0] + strides[1] - 1) / strides[1];  // the slice number on the width axis

    mlir::SmallVector<mlir::Value> wSliced;
    for (auto i : irange(stepsW)) {  // slicing on the horizontal axis
        mlir::SmallVector<mlir::Value> hSliced;
        for (auto j : irange(stepsH)) {  // slicing on the vertical axis
            Shape offsets(inputShape.size());
            SmallVector<int64_t> slicePaddingEnd(2);

            // calculate the offset for the slice
            offsets[Dims4D::Act::H] = j * minStride;
            offsets[Dims4D::Act::W] = i * minStride;
            if (inputShape[Dims4D::Act::H] <= offsets[Dims4D::Act::H] ||
                inputShape[Dims4D::Act::W] <= offsets[Dims4D::Act::W]) {
                continue;
            }

            // calculate the shape of the slice
            SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                            inputShape[Dims4D::Act::H] - offsets[Dims4D::Act::H],
                                            inputShape[Dims4D::Act::W] - offsets[Dims4D::Act::W]};

            const auto loc = appendLoc(origOp->getLoc(), "slice {0}, {1}", i, j);

            auto slicedInput = rewriter.create<IE::SliceOp>(
                    loc, origOp->getOperand(0), getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            // create avg pooling for this slice with new symmetric stride
            auto roundingTypeAttr = IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR);
            auto newOp = rewriter.create<IE::AvgPoolOp>(loc, slicedInput.result(), origOp.kernel_size(), newStrides,
                                                        origOp.pads_begin(), origOp.pads_end(), roundingTypeAttr,
                                                        origOp.exclude_padsAttr(), origOp.post_opAttr());
            hSliced.push_back(newOp->getResult(0));
        }
        if (!hSliced.empty()) {
            // concatenate the slices if there are more than one slice on vertical axis, and store it in wSliced
            wSliced.push_back(hSliced.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), hSliced,
                                                                                  Dims4D::Act::H, 1, stepsH)
                                                  : hSliced.front());
        }
    }
    if (wSliced.empty()) {
        return errorAt(origOp->getLoc(), "Empty slice for avgpool");
    }

    // concatenate the slices if there are more than one slice on horizontal axis
    const auto concatOp = wSliced.size() != 1
                                  ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), wSliced, Dims4D::Act::W, 1, stepsW)
                                  : wSliced.front();
    rewriter.replaceOp(origOp, concatOp);
    return concatOp;
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("MaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    calculateKernelsAndPadding(kernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel, false, _log.nest(2));
    mlir::MLIRContext* ctx = origOp->getContext();

    const auto origStridesAttr = origOp.strides();
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    std::array<int64_t, 4> origPadding = {padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]};
    std::array<int64_t, 4> inputPadding = calculatedPadding;

    auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    const auto origStrides = parseIntArrayAttr<int64_t>(origStridesAttr);

    auto outShape = getShape(origOp.output());
    const auto inputShape = getShape(origOp.input());

    // Usually this kind of maxpool is converted from ReduceMin/ReduceMax
    // It has a large kernel to reduce one of the axis dim to 1, and strides are (1, 1)
    // Padding of MaxPool which convert from ReduceMin/ReduceMax are all zeros
    const auto SX = origStrides[Dims4D::Strides::X.ind()];
    const auto SY = origStrides[Dims4D::Strides::Y.ind()];
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    auto isZero = [](auto val) {
        return val == 0;
    };
    bool axisIsReducedToOneAndStrideIsOne =
            ((inputShape[Dims4D::Act::H] == KY && SY == 1) || (inputShape[Dims4D::Act::W] == KX && SX == 1)) &&
            llvm::all_of(padsBegin, isZero) && llvm::all_of(padsEnd, isZero);

    auto isPrime = [](int64_t n) -> bool {
        if (n <= 1)
            return false;
        for (int64_t i = 2; i * i <= n; i++)
            if (n % i == 0)
                return false;
        return true;
    };

    bool ifActXNeedPadding =
            (KX > VPU::NCEInvariant::MAX_KERNEL_SIZE && isPrime(KX)) && axisIsReducedToOneAndStrideIsOne;
    bool ifActYNeedPadding =
            (KY > VPU::NCEInvariant::MAX_KERNEL_SIZE && isPrime(KY)) && axisIsReducedToOneAndStrideIsOne;

    // in this case stride shall be taken into account and pyramid cascading does not work
    // use expression orig_kernel = sum (k1, k2, ..., ki)
    bool ifSequenceOpCannotPyramidCascaded = (SX != KX || SY != KY) &&
                                             !(outShape[Dims4D::Act::H] == 1 && outShape[Dims4D::Act::W] == 1) &&
                                             !axisIsReducedToOneAndStrideIsOne;

    if (ifSequenceOpCannotPyramidCascaded) {
        inputPadding = origPadding;
        stridesAttr = origStridesAttr;
    }

    mlir::ArrayAttr firstOpPadBegin, firstOpPadEnd;
    bool unsuportedPad = false;
    bool isSupportedYPadding = (inputPadding[0] < firstOpKernel[Dims4D::Kernel::Y.ind()] / 2) &&
                               (inputPadding[1] < firstOpKernel[Dims4D::Kernel::Y.ind()] / 2);
    bool isSupportedXPadding = (inputPadding[2] < firstOpKernel[Dims4D::Kernel::X.ind()] / 2) &&
                               (inputPadding[3] < firstOpKernel[Dims4D::Kernel::X.ind()] / 2);
    bool allPaddingsEqual = std::all_of(inputPadding.cbegin(), inputPadding.cend(), [&inputPadding](int64_t inPad) {
        return inPad == inputPadding[0];
    });

    if (!isSupportedXPadding && !isSupportedYPadding && allPaddingsEqual) {
        unsuportedPad = true;
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({inputPadding[2] / 2, inputPadding[0] / 2}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({inputPadding[3] / 2, inputPadding[1] / 2}));
    } else {
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({inputPadding[2], inputPadding[0]}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({inputPadding[3], inputPadding[1]}));
    }
    const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    auto strides = parseIntArrayAttr<int64_t>(stridesAttr);
    stridesAttr = getGlobalOrOrigStride(ctx, origOp.input(), firstOpKernel,
                                        {strides[Dims4D::Strides::Y.ind()], strides[Dims4D::Strides::X.ind()]},
                                        firstOpPadBegin, firstOpPadEnd);
    if (axisIsReducedToOneAndStrideIsOne) {
        stridesAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    }
    strides = parseIntArrayAttr<int64_t>(stridesAttr);
    auto firstOpInput = origOp.input();
    // When the kernel size bigger than MAX_KERNEL_SIZE and it's prime value, need extra padding.
    // case1: tensor<1, 1, 71, 2> -> tensor<1, 1, 1, 2>, kernel[71, 1], need extra padding
    // case2: tensor<1, 1, 32, 2> -> tensor<1, 1, 1, 2>, kernel[32, 1], no padding needed
    // For MaxPool, if the value in activation is negative, padding zero will cause AC issue(unlike AvgPool).
    // So we shouldn't pad zero valuses directly and here we slice the value from the original activations.
    auto paddingActivation = [&](SmallVector<int64_t> sliceShape, vpux::Dim dimsValue) {
        Shape offsets = Shape(inputShape.size(), 0);
        auto slicedInput =
                rewriter.create<IE::SliceOp>(origOp.getLoc(), origOp.input(), getIntArrayAttr(ctx, offsets.raw()),
                                             getIntArrayAttr(ctx, sliceShape))
                        .result();
        SmallVector<mlir::Value> concatInput;
        concatInput.push_back(firstOpInput);
        concatInput.push_back(slicedInput);
        firstOpInput = rewriter.create<IE::ConcatOp>(origOp.getLoc(), concatInput, dimsValue).output();
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({0, 0}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({0, 0}));
    };

    if (ifActXNeedPadding) {
        SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                        inputShape[Dims4D::Act::H], calculatedPadding[0] + calculatedPadding[1]};
        paddingActivation(sliceShape, Dims4D::Act::W);
    }
    if (ifActYNeedPadding) {
        SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                        calculatedPadding[2] + calculatedPadding[3], inputShape[Dims4D::Act::W]};
        paddingActivation(sliceShape, Dims4D::Act::H);
    }

    auto firstOp = rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), firstOpInput, firstOpKernelAttr, stridesAttr,
                                                  firstOpPadBegin, firstOpPadEnd, origOp.rounding_type(),
                                                  origOp.post_opAttr());
    stridesAttr = sequencedOpKernelAttr;

    if (ifSequenceOpCannotPyramidCascaded) {
        sequencedOpKernel[Dims4D::Kernel::X.ind()] = KX - firstOpKernel[Dims4D::Kernel::X.ind()] + 1;
        sequencedOpKernel[Dims4D::Kernel::Y.ind()] = KY - firstOpKernel[Dims4D::Kernel::Y.ind()] + 1;
        stridesAttr = origStridesAttr;
    }
    if (axisIsReducedToOneAndStrideIsOne || ifSequenceOpCannotPyramidCascaded) {
        calculatedPadding = {0, 0, 0, 0};
    }
    if (unsuportedPad) {
        calculatedPadding[0] = inputPadding[0] - inputPadding[0] / 2;
        calculatedPadding[1] = inputPadding[1] - inputPadding[1] / 2;
        calculatedPadding[2] = inputPadding[2] - inputPadding[2] / 2;
        calculatedPadding[3] = inputPadding[3] - inputPadding[3] / 2;
    }

    const auto sequencedOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto sequencedOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
    sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    strides = parseIntArrayAttr<int64_t>(stridesAttr);
    stridesAttr = getGlobalOrOrigStride(ctx, firstOp.output(), sequencedOpKernel,
                                        {strides[Dims4D::Strides::Y.ind()], strides[Dims4D::Strides::X.ind()]},
                                        sequencedOpPadBegin, sequencedOpPadEnd);
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(origOp, firstOp.output(), sequencedOpKernelAttr, stridesAttr,
                                               sequencedOpPadBegin, sequencedOpPadEnd, origOp.rounding_type(),
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

mlir::Value getZerosConst(mlir::PatternRewriter& rewriter, Shape constShape, IE::ConvolutionOp origOp) {
    const auto elemType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(constShape), elemType);

    mlir::DenseElementsAttr denseElementVal = wrapData(dataStorageType, 0.0f);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "ConvolutionOp has incompatible data type {0}, only float16 or float32 are supported", elemType);

    return rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, Const::ContentAttr::get(denseElementVal))
            .output();
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

    const auto hasSupportedKernels = [](const SmallVector<int64_t>& kernelSize) {
        const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
        const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

        return KY <= VPU::NCEInvariant::MAX_KERNEL_SIZE && KX <= VPU::NCEInvariant::MAX_KERNEL_SIZE;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<Const::DeclareOp>();

    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        if (hasSupportedKernels(kernelSize)) {
            return true;
        }

        auto unsupportedKernelCheck = [&](int32_t kernelInd) {
            const auto maxKernelSizeSupported =
                    VPU::NCEInvariant::MAX_KERNEL_SIZE *
                    VPU::NCEInvariant::MAX_KERNEL_SIZE;  // we can only get 2 factors
                                                         // and max kernel should be 11 * 11 = 121
            return (kernelSize[kernelInd] > maxKernelSizeSupported);
        };

        if (unsupportedKernelCheck(Dims4D::Kernel::X.ind())) {
            _log.trace("Unsupported MaxPool kernel width dimension '{0}'", kernelSize[Dims4D::Kernel::X.ind()]);
            return true;
        }
        if (unsupportedKernelCheck(Dims4D::Kernel::Y.ind())) {
            _log.trace("Unsupported MaxPool kernel height dimension '{0}'", kernelSize[Dims4D::Kernel::Y.ind()]);
            return true;
        }
        return false;
    });
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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MaxPoolRewriter>(&ctx, _log);
    patterns.add<ConvRewriter>(&ctx, _log);

    mlir::RewritePatternSet avgPoolPatterns(&ctx);
    avgPoolPatterns.add<AveragePoolRewriter>(&ctx, _log);

    auto func = getOperation();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
    // For AvgPool, each execution of matchAndRewrite will only split it into two AvgPools. In some cases, the split
    // AvgPool will need to continue splitting until all the split AvgPools are legal. So used GreedyRewriteConfig here.
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(avgPoolPatterns),
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
