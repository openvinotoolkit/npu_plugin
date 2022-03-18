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
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
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

constexpr int64_t MAX_STRIDE = 8;

struct OperationPart {
    mlir::ArrayAttr strides;
    mlir::ArrayAttr padBegin;
    mlir::ArrayAttr padEnd;
};

mlir::Operation* createFQ(mlir::PatternRewriter& rewriter, mlir::Operation* inputOp, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.output().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(inputOp->getResult(0)));
    return rewriter.create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, inputOp->getResult(0), fq.input_low(),
                                               fq.input_high(), fq.output_low(), fq.output_high(), fq.levels(),
                                               fq.auto_broadcast());
}

/*
    This splitter will slice the op into output_height x output_width ops of the same type with a new stride of 1 x 1.

    Each resulting op will receive a slice of the original input with
        slice_width = kernel_width and slice_height = kernel_height.

    The results of the new ops will have output_height = 1 and output_width = 1 and
    they will be concatenated, first over W axis and then over H axis.
*/
mlir::LogicalResult generalSplitter(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                    mlir::ArrayAttr stridesAttr, ArrayRef<int64_t> kernelSize,
                                    mlir::ArrayAttr padBeginAttr, mlir::ArrayAttr padEndAttr,
                                    FuncRef<mlir::Operation*(mlir::Location, mlir::Value, OperationPart)> makeOperation,
                                    Logger log) {
    mlir::MLIRContext* ctx = origOp->getContext();

    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr);

    const auto KY = kernelSize[1];
    const auto KX = kernelSize[0];

    const auto inputShape = getShape(origOp->getOperand(0));
    const auto H = inputShape[Dims4D::Act::H];
    const auto W = inputShape[Dims4D::Act::W];

    const SmallVector<int64_t> newStrides = {1, 1};

    log.trace("New strides for {0} are {1}", origOp->getLoc(), newStrides);

    const auto padBegin = parseIntArrayAttr<int64_t>(padBeginAttr);
    const auto padEnd = parseIntArrayAttr<int64_t>(padEndAttr);

    const int64_t sliceH = (H + padBegin[0] + padEnd[0] - KY) / strides[0] + 1;
    const int64_t sliceW = (W + padBegin[1] + padEnd[1] - KX) / strides[1] + 1;

    const auto inputFQ = origOp->getOperand(0).getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp->getResult(0).user_begin()));

    Shape offsets(inputShape.size());
    SmallVector<mlir::Value> hSliced;
    for (auto i : irange(sliceH)) {
        const auto padTop = (i == 0) ? padBegin[0] : 0;
        const auto padBottom = (i + 1 == sliceH) ? padEnd[0] : 0;

        offsets[Dims4D::Act::W] = 0;

        SmallVector<mlir::Value> wSliced;
        for (auto j : irange(sliceW)) {
            const auto padLeft = (j == 0) ? padBegin[1] : 0;
            const auto padRight = (j + 1 == sliceW) ? padEnd[1] : 0;

            const SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                                  KY - padTop - padBottom, KX - padLeft - padRight};

            const auto sliceName = llvm::formatv("slice {0}, {1}", i, j).str();
            const auto loc = appendLoc(origOp->getLoc(), sliceName);

            mlir::Operation* slicedInput = rewriter.create<IE::SliceOp>(
                    loc, origOp->getOperand(0), getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            // TODO: temporary FQ propagation
            if (inputFQ != nullptr) {
                slicedInput = createFQ(rewriter, slicedInput, inputFQ);
            }

            const SmallVector<int64_t> newPadBegin = {padTop, padLeft};
            const SmallVector<int64_t> newPadEnd = {padBottom, padRight};

            auto* newOp = makeOperation(loc, slicedInput->getResult(0),
                                        {getIntArrayAttr(ctx, newStrides), getIntArrayAttr(ctx, newPadBegin),
                                         getIntArrayAttr(ctx, newPadEnd)});
            wSliced.push_back(newOp->getResult(0));

            // TODO: temporary FQ propagation
            if (outputFQ != nullptr) {
                newOp = createFQ(rewriter, newOp, outputFQ);
            }

            offsets[Dims4D::Act::W] += strides[1] - padLeft;
        }

        offsets[Dims4D::Act::H] += strides[0] - padTop;

        if (!wSliced.empty()) {
            hSliced.push_back(wSliced.size() != 1
                                      ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), wSliced, Dims4D::Act::W)
                                      : wSliced.front());
        }
    }

    if (!hSliced.empty()) {
        if (hSliced.size() == 1) {
            rewriter.replaceOp(origOp, hSliced.front());
        } else {
            rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, hSliced, Dims4D::Act::H);
        }
    } else {
        return mlir::failure();
    }

    return mlir::success();
}

//
// ConvGeneralRewriter
//

class ConvGeneralRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvGeneralRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, vpux::benefitLow), _log(log) {
        setDebugName("ConvGeneralRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvGeneralRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto filterShape = getShape(origOp.filter());

    return generalSplitter(
            origOp, rewriter, origOp.strides(),
            makeArrayRef({filterShape[Dims4D::Filter::KX], filterShape[Dims4D::Filter::KY]}), origOp.pads_begin(),
            origOp.pads_end(),
            [&](mlir::Location loc, mlir::Value input, OperationPart part) -> mlir::Operation* {
                return rewriter.create<IE::ConvolutionOp>(loc, input, origOp.filter(), origOp.bias(), part.strides,
                                                          part.padBegin, part.padEnd, origOp.dilations(),
                                                          origOp.post_opAttr());
            },
            _log.nest());
}

//
// GroupConvGeneralRewriter
//

class GroupConvGeneralRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvGeneralRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, vpux::benefitLow), _log(log) {
        setDebugName("GroupConvGeneralRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupConvGeneralRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution layer layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto filterShape = getShape(origOp.filter());

    return generalSplitter(
            origOp, rewriter, origOp.strides(),
            makeArrayRef({filterShape[Dims4D::Filter::KX], filterShape[Dims4D::Filter::KY]}), origOp.pads_begin(),
            origOp.pads_end(),
            [&](mlir::Location loc, mlir::Value input, OperationPart part) -> mlir::Operation* {
                return rewriter.create<IE::GroupConvolutionOp>(loc, input, origOp.filter(), origOp.bias(), part.strides,
                                                               part.padBegin, part.padEnd, origOp.dilations(),
                                                               origOp.groupsAttr(), origOp.post_opAttr());
            },
            _log.nest());
}

//
// MaxPoolGeneralRewriter
//

class MaxPoolGeneralRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolGeneralRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx, vpux::benefitLow), _log(log) {
        setDebugName("MaxPoolGeneralRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolGeneralRewriter::matchAndRewrite(IE::MaxPoolOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto kernelSz = parseIntArrayAttr<int64_t>(origOp.kernel_size());

    return generalSplitter(
            origOp, rewriter, origOp.strides(), makeArrayRef({kernelSz[1], kernelSz[0]}), origOp.pads_begin(),
            origOp.pads_end(),
            [&](mlir::Location loc, mlir::Value input, OperationPart part) -> mlir::Operation* {
                return rewriter.create<IE::MaxPoolOp>(loc, input, origOp.kernel_size(), part.strides, part.padBegin,
                                                      part.padEnd, origOp.rounding_type(), origOp.post_opAttr());
            },
            _log.nest());
}

/*
    This transformation takes the original op with stride > MAX_STRIDE and replaces it with
    an op of the same type with smaller stride and a downsampling max pool afterward.

    The stride for the new op is chosen such that new_stride <= MAX_STRIDE and orig_stride % new_stride == 0.
    The new max pool op will have kernel size of 1 x 1 and stride = orig_stride / new_stride.
    The purpose of the max pool op is to downsample the larger output obtained by striding the original op
    with a smaller stride.

    The replacement is applied even if only one of the strides is larger than MAX_STRIDE. The other stride
    will stay the same, while the corresponding max pool stride will be 1.

    This optimization cannot be applied if the large strides are not multiples of a number smaller than MAX_STRIDE
    (e.g. if old_stride is a prime number).
*/
mlir::LogicalResult opWithMaxPoolOptimization(
        mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::ArrayAttr stridesAttr,
        FuncRef<mlir::Operation*(mlir::Location, mlir::Value, mlir::ArrayAttr)> makeOperation, Logger log) {
    mlir::MLIRContext* ctx = origOp->getContext();

    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr);
    const auto SY = strides[0];
    const auto SX = strides[1];

    auto getSmallerStride = [](const int64_t stride) -> int64_t {
        for (int64_t k = MAX_STRIDE; k > 1; --k) {
            if (stride % k == 0)
                return k;
        }

        return stride;
    };

    const SmallVector<int64_t> newStride = {(SY > MAX_STRIDE) ? getSmallerStride(SY) : SY,
                                            (SX > MAX_STRIDE) ? getSmallerStride(SX) : SX};

    if (newStride[0] > MAX_STRIDE || newStride[1] > MAX_STRIDE)
        return mlir::failure();

    log.trace("New strides for {0} are {1}", origOp->getLoc(), newStride);

    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp->getResult(0).user_begin()));

    auto* newOp = makeOperation(appendLoc(origOp->getLoc(), "small-stride"), origOp->getOperand(0),
                                getIntArrayAttr(ctx, newStride));

    // TODO: temporary FQ propagation
    if (outputFQ != nullptr) {
        newOp = createFQ(rewriter, newOp, outputFQ);
    }

    const SmallVector<int64_t> maxPoolStrides = {SY / newStride[0], SX / newStride[1]};
    const SmallVector<int64_t> maxPoolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padsAttr = getIntArrayAttr(ctx, pads);
    const auto loc = appendLoc(origOp->getLoc(), "_maxpool_downsample");

    log.trace("MaxPool {0} strides are {1}", loc, newStride);
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            origOp, newOp->getResult(0), getIntArrayAttr(ctx, maxPoolKernels), getIntArrayAttr(ctx, maxPoolStrides),
            padsAttr, padsAttr, vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), nullptr);

    return mlir::success();
}

//
// ConvMPOptimizationRewriter
//

class ConvMPOptimizationRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvMPOptimizationRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, vpux::benefitHigh), _log(log) {
        setDebugName("ConvMPOptimizationRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvMPOptimizationRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    return opWithMaxPoolOptimization(
            origOp, rewriter, origOp.strides(),
            [&](mlir::Location loc, mlir::Value input, mlir::ArrayAttr strides) -> mlir::Operation* {
                return rewriter.create<IE::ConvolutionOp>(loc, input, origOp.filter(), origOp.bias(), strides,
                                                          origOp.pads_begin(), origOp.pads_end(), origOp.dilations(),
                                                          origOp.post_opAttr());
            },
            _log.nest());
}

//
// GroupConvMPOptimizationRewriter
//

class GroupConvMPOptimizationRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvMPOptimizationRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, vpux::benefitHigh), _log(log) {
        setDebugName("GroupConvMPOptimizationRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupConvMPOptimizationRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution layer layer at '{1}'", getDebugName(), origOp->getLoc());

    return opWithMaxPoolOptimization(
            origOp, rewriter, origOp.strides(),
            [&](mlir::Location loc, mlir::Value input, mlir::ArrayAttr strides) -> mlir::Operation* {
                return rewriter.create<IE::GroupConvolutionOp>(
                        loc, input, origOp.filter(), origOp.bias(), strides, origOp.pads_begin(), origOp.pads_end(),
                        origOp.dilations(), origOp.groupsAttr(), origOp.post_opAttr());
            },
            _log.nest());
}

//
// MaxPoolMPOptimizationRewriter
//

class MaxPoolMPOptimizationRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolMPOptimizationRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx, vpux::benefitMid), _log(log) {
        setDebugName("MaxPoolMPOptimizationRewriter");
        setHasBoundedRewriteRecursion();
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolMPOptimizationRewriter::matchAndRewrite(IE::MaxPoolOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    return opWithMaxPoolOptimization(
            origOp, rewriter, origOp.strides(),
            [&](mlir::Location loc, mlir::Value input, mlir::ArrayAttr strides) -> mlir::Operation* {
                return rewriter.create<IE::MaxPoolOp>(loc, input, origOp.kernel_size(), strides, origOp.pads_begin(),
                                                      origOp.pads_end(), origOp.rounding_type(), origOp.post_opAttr());
            },
            _log.nest());
}

//
// HandleLargeStridesPass
//

class HandleLargeStridesPass final : public IE::HandleLargeStridesBase<HandleLargeStridesPass> {
public:
    explicit HandleLargeStridesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleLargeStridesPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasSupportedStrides = [](const SmallVector<int64_t>& strides) {
        const auto SY = strides[0];
        const auto SX = strides[1];

        return SY <= MAX_STRIDE && SX <= MAX_STRIDE;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(op.strides());
        return hasSupportedStrides(kernelStrides);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(op.strides());
        return hasSupportedStrides(kernelStrides);
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        const auto kernelStrides = parseIntArrayAttr<int64_t>(op.strides());
        return hasSupportedStrides(kernelStrides);
    });

    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvMPOptimizationRewriter>(&ctx, _log);
    patterns.insert<GroupConvMPOptimizationRewriter>(&ctx, _log);
    patterns.insert<MaxPoolMPOptimizationRewriter>(&ctx, _log);
    patterns.insert<ConvGeneralRewriter>(&ctx, _log);
    patterns.insert<GroupConvGeneralRewriter>(&ctx, _log);
    patterns.insert<MaxPoolGeneralRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createHandleLargeStridesPass(Logger log) {
    return std::make_unique<HandleLargeStridesPass>(log);
}
