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

void getFactorsForSecondDimension(std::array<int64_t, 4>& padding, std::array<int64_t, 2>& firstOpKernel,
                                  std::array<int64_t, 2>& sequencedOpKernel, int32_t smallDim, Logger log,
                                  ArrayRef<int64_t> kernelSize) {
    const auto factorsSecondDim = vpux::IE::getFactors(kernelSize[smallDim]);  // toggling between the two kernel sizes
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

void calculateKernelsAndPadding(ArrayRef<int64_t> kernelSize, std::array<int64_t, 4>& padding,
                                std::array<int64_t, 2>& firstOpKernel, std::array<int64_t, 2>& sequencedOpKernel,
                                Logger log) {
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
    const auto factors = vpux::IE::getFactors(largerKernelSize);

    log.trace("Large Dimension kernelSize[{0}] = {1}, larger factor: {2} , smaller factor: {3}", largeDim,
              largerKernelSize, factors.larger, factors.smaller);
    VPUX_THROW_UNLESS((factors.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE) &&
                              (factors.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE),
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
            getFactorsForSecondDimension(padding, firstOpKernel, sequencedOpKernel, smallDim, log, kernelSize);
        } else {
            firstOpKernel[smallDim] = kernelSize[smallDim];
            sequencedOpKernel[smallDim] =
                    1;  // the smallDim was not factorized, the multiplication kSize*1 covers the second op

            padding[PADDING_BOT] = 0;
        }
        // factors multiplied > kernel, we need padding
        padding[PADDING_RIGHT] = (multipliedFactors > kernelSize[largeDim]) ? 1 : 0;

        if (largeDim != Dims4D::Kernel::X.ind()) {
            // change the padding on the other dimensions as largeDim was not on the width dimension - PADD_RIGHT
            std::swap(padding[PADDING_RIGHT], padding[PADDING_BOT]);
        }
    } else {
        firstOpKernel[smallDim] = factors.larger;  // largeDim has the same kernel size as smallDim
        sequencedOpKernel[smallDim] = factors.smaller;
        padding[PADDING_RIGHT] = padding[PADDING_BOT] = (multipliedFactors > kernelSize[largeDim]) ? 1 : 0;
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

    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    calculateKernelsAndPadding(kernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel, _log.nest(2));

    auto* ctx = origOp->getContext();

    const auto firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));

    const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    const auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    auto firstOp = rewriter.create<IE::AvgPoolOp>(origOp->getLoc(), origOp.input(), firstOpKernelAttr,
                                                  firstOpKernelAttr, firstOpPadBegin, firstOpPadEnd,
                                                  origOp.rounding_typeAttr(), origOp.exclude_padsAttr());

    const auto firstOpOutputShapeType = firstOp.output().getType().cast<vpux::NDTypeInterface>();
    const auto firstOpOutputShape = firstOpOutputShapeType.getShape().raw();

    auto firstAvgPoolOutput = firstOp.output();
    auto checkStrideRelation = [](const int64_t strideLeft, const int64_t strideRight) -> bool {
        return strideLeft > strideRight && strideLeft % strideRight == 0;
    };

    bool useSplitAvgOperationSlicing =
            checkStrideRelation(firstOpKernel[Dims4D::Strides::X.ind()], firstOpKernel[Dims4D::Strides::Y.ind()]) ||
            checkStrideRelation(firstOpKernel[Dims4D::Strides::Y.ind()], firstOpKernel[Dims4D::Strides::X.ind()]);
    if (useSplitAvgOperationSlicing) {
        const auto concatOp = splitAvgOperationSlicing(firstOp, rewriter);
        if (mlir::failed(concatOp)) {
            return mlir::failure();
        }
        firstAvgPoolOutput = concatOp.getValue();
    }

    auto globalAvgOverH = firstOpOutputShape[Dims4D::Act::H.ind()] == sequencedOpKernel[0];
    auto globalAvgOverW = firstOpOutputShape[Dims4D::Act::W.ind()] == sequencedOpKernel[1];
    std::array<int64_t, 2> sequencedOpStrides = {1, 1};
    if (!globalAvgOverH) {
        sequencedOpStrides[0] = sequencedOpKernel[0];
    }
    if (!globalAvgOverW) {
        sequencedOpStrides[1] = sequencedOpKernel[1];
    }
    const auto sequencedOpStridesAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpStrides));

    calculatedPadding = {0, 0, 0, 0};
    const auto sequencedOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto sequencedOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(origOp, origOp.getType(), firstAvgPoolOutput, sequencedOpKernelAttr,
                                               sequencedOpStridesAttr, sequencedOpPadBegin, sequencedOpPadEnd,
                                               origOp.rounding_typeAttr(), origOp.exclude_padsAttr());

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

            const auto sliceName = llvm::formatv("slice {0}, {1}", i, j).str();
            const auto loc = appendLoc(origOp->getLoc(), sliceName);

            auto slicedInput = rewriter.create<IE::SliceOp>(
                    loc, origOp->getOperand(0), getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            // create avg pooling for this slice with new symmetric stride
            auto newOp = rewriter.create<IE::AvgPoolOp>(loc, slicedInput.result(), origOp.kernel_size(), newStrides,
                                                        origOp.pads_begin(), origOp.pads_end(),
                                                        origOp.rounding_typeAttr(), origOp.exclude_padsAttr());
            hSliced.push_back(newOp->getResult(0));
        }
        if (!hSliced.empty()) {
            // concatenate the slices if there are more than one slice on vertical axis, and store it in wSliced
            wSliced.push_back(hSliced.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), hSliced,
                                                                                  Dims4D::Act::H, minStride, maxStride)
                                                  : hSliced.front());
        }
    }
    if (wSliced.empty()) {
        return errorAt(origOp->getLoc(), "Empty slice for avgpool");
    }

    // concatenate the slices if there are more than one slice on horizontal axis
    const auto concatOp = wSliced.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), wSliced, Dims4D::Act::W,
                                                                              minStride, maxStride)
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
    calculateKernelsAndPadding(kernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel, _log.nest(2));

    mlir::MLIRContext* ctx = origOp->getContext();

    const auto origStridesAttr = origOp.strides();
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    std::array<int64_t, 4> origPadding = {padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]};
    std::array<int64_t, 4> inputPadding = calculatedPadding;

    auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    const auto origStrides = parseIntArrayAttr<int64_t>(origStridesAttr);

    if (origStrides[Dims4D::Strides::X.ind()] != kernelSize[Dims4D::Kernel::X.ind()] ||
        origStrides[Dims4D::Strides::Y.ind()] != kernelSize[Dims4D::Kernel::Y.ind()]) {
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

    auto firstOp = rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), origOp.input(), firstOpKernelAttr, stridesAttr,
                                                  firstOpPadBegin, firstOpPadEnd, origOp.rounding_type(),
                                                  origOp.post_opAttr());

    stridesAttr = sequencedOpKernelAttr;

    if (origStrides[Dims4D::Strides::X.ind()] != kernelSize[Dims4D::Kernel::X.ind()] ||
        origStrides[Dims4D::Strides::Y.ind()] != kernelSize[Dims4D::Kernel::Y.ind()]) {
        // in this case stride shall be taken into account and pyramid cascading does not work
        // use expression orig_kernel = sum (k1, k2, ..., ki)
        sequencedOpKernel[Dims4D::Kernel::X.ind()] =
                kernelSize[Dims4D::Kernel::X.ind()] - firstOpKernel[Dims4D::Kernel::X.ind()] + 1;
        sequencedOpKernel[Dims4D::Kernel::Y.ind()] =
                kernelSize[Dims4D::Kernel::Y.ind()] - firstOpKernel[Dims4D::Kernel::Y.ind()] + 1;
        stridesAttr = origStridesAttr;
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
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(origOp, firstOp.output(), sequencedOpKernelAttr, stridesAttr,
                                               sequencedOpPadBegin, sequencedOpPadEnd, origOp.rounding_type(),
                                               origOp.post_opAttr());
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
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        if (hasSupportedKernels(kernelSize)) {
            return true;
        }
        const auto inDataType = op.input().getType().cast<vpux::NDTypeInterface>();
        const auto inDataShape = inDataType.getShape().raw();
        const auto strides = parseIntArrayAttr<int64_t>(op.strides());
        const auto maxKernelSizeSupported =
                VPU::NCEInvariant::MAX_KERNEL_SIZE *
                VPU::NCEInvariant::MAX_KERNEL_SIZE;  // we can only get 2 factors and max kernel should be 11 * 11 = 121
        auto unsupportedKernelCheck = [&](int32_t kernelInd, int32_t actInd, int32_t strideInd) {
            return ((kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]) ||
                    kernelSize[kernelInd] > maxKernelSizeSupported);
        };

        if (unsupportedKernelCheck(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind())) {
            _log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
            return true;
        }
        if (unsupportedKernelCheck(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
            _log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
            return true;
        }
        return false;
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        return hasSupportedKernels(kernelSize);
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<AveragePoolRewriter>(&ctx, _log);
    patterns.insert<MaxPoolRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
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
