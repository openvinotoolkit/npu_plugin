//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// DilatedConvolutionRewriter
//

class DilatedConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    DilatedConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("DilatedConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// here we have a optimization for some special case when Y dilation is 1 or X dilation is 1
// assume Y dilation is 1, kernel is 2*2, we slice kernel to 2 2*1(Y*X), and slice the input
// accordingly, use eltwise to add the two outputs, then this is the first pixel of X, use
// the same way to get other X and then concat all the results

// step1: for each pixel of output W, we convert
//
//      [act]        [w]                     [act]       [w]         [act]        [w]
//        |           |           to           |          |            |           |
//       -(dilatedConv)-                    (slice)    (slice)      (slice)     (slice)
//                                             |          |            |           |
//                                               -(conv)-                -(conv)-
//                                                   |                       |
//                                                     ---- (eltwise) -----
// step2: then use concat to concat each pixel of W

mlir::LogicalResult DilatedConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());
    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.dilations()));
    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.pads_begin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.pads_end()));
    const auto outputShape = getShape(origOp);
    const auto outH = outputShape[Dims4D::Act::H];
    const auto outW = outputShape[Dims4D::Act::W];
    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto expandKernelX = (KX - 1) * dilations[Dims4D::Dilation::X] + 1;
    const auto expandKernelY = (KY - 1) * dilations[Dims4D::Dilation::Y] + 1;

    if ((expandKernelX > vpux::VPU::NCEInvariant::MAX_KERNEL_SIZE && dilations[Dims4D::Dilation::Y] == 1 &&
         padStart[Dims4D::PadsBegin::Left] == 0 && padEnd[Dims4D::PadsEnd::Right] == 0) ||
        (expandKernelY > vpux::VPU::NCEInvariant::MAX_KERNEL_SIZE && dilations[Dims4D::Dilation::X] == 1 &&
         padStart[Dims4D::PadsBegin::Top] == 0 && padEnd[Dims4D::PadsEnd::Bottom] == 0)) {
        _log.trace("[{0}] Slice Dilated conv to small task '{1}'", getDebugName(), origOp->getLoc());

        mlir::MLIRContext* ctx = origOp->getContext();
        const auto inputShape = getShape(origOp->getOperand(0));
        const auto IC = filterShape[Dims4D::Filter::IC];
        const auto OC = filterShape[Dims4D::Filter::OC];
        const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
        const auto broadcastType =
                vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
        bool isDilatedX = dilations[Dims4D::Dilation::Y] == 1 ? true : false;

        mlir::SmallVector<mlir::Value> slicedFilters;
        mlir::SmallVector<mlir::Value> concats;
        int64_t kernel = isDilatedX ? KX : KY;
        for (int64_t k = 0; k < kernel; k++) {
            SmallVector<int64_t> sliceShape{OC, IC, isDilatedX ? KY : 1, isDilatedX ? 1 : KX};
            Shape offsets(filterShape.size());
            offsets[Dims4D::Filter::KX] = isDilatedX ? k : offsets[Dims4D::Filter::KX];
            offsets[Dims4D::Filter::KY] = isDilatedX ? offsets[Dims4D::Filter::KY] : k;
            auto slice =
                    rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.filter(), getIntArrayAttr(ctx, offsets.raw()),
                                                 getIntArrayAttr(ctx, sliceShape));
            slicedFilters.push_back(slice);
        }

        llvm::MapVector<int64_t, mlir::Value> slicedInputs;
        int64_t outWOrH = isDilatedX ? outW : outH;
        for (int64_t i = 0; i < outWOrH; i++) {
            mlir::SmallVector<mlir::Value> eltwises;

            for (int64_t k = 0; k < kernel; k++) {
                int64_t startW =
                        isDilatedX ? (i * strides[Dims4D::Strides::X] + k * dilations[Dims4D::Dilation::X]) : 0;
                VPUX_THROW_WHEN(startW >= inputShape[Dims4D::Act::W], "dimension W out of range");
                int64_t startH =
                        isDilatedX ? 0 : (i * strides[Dims4D::Strides::Y] + k * dilations[Dims4D::Dilation::Y]);
                VPUX_THROW_WHEN(startH >= inputShape[Dims4D::Act::H], "dimension H out of range");
                int64_t processingHOrW = isDilatedX ? startW : startH;

                mlir::Value convInput;
                // check if the input has been already sliced
                if (slicedInputs.find(processingHOrW) != slicedInputs.end()) {
                    convInput = slicedInputs[processingHOrW];
                } else {
                    SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                                    isDilatedX ? inputShape[Dims4D::Act::H] : 1,
                                                    isDilatedX ? 1 : inputShape[Dims4D::Act::W]};
                    Shape offsets(inputShape.size());
                    offsets[Dims4D::Act::W] = startW;
                    offsets[Dims4D::Act::H] = startH;
                    convInput = rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.input(),
                                                             getIntArrayAttr(ctx, offsets.raw()),
                                                             getIntArrayAttr(ctx, sliceShape));
                    slicedInputs.insert({processingHOrW, convInput});
                }
                // add bias and post process for the last convolution and eltwise.
                auto conv = rewriter.create<IE::ConvolutionOp>(
                        origOp->getLoc(), convInput, slicedFilters[k], (k == (kernel - 1)) ? origOp.bias() : nullptr,
                        origOp.strides(), origOp.pads_begin(), origOp.pads_end(),
                        getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})),
                        (kernel == 1) ? origOp.post_opAttr() : nullptr);

                if (eltwises.size() > 0) {
                    auto add = rewriter.create<IE::AddOp>(origOp->getLoc(), eltwises.back(), conv, broadcastType,
                                                          (k == (kernel - 1)) ? origOp.post_opAttr() : nullptr);
                    eltwises.push_back(add);
                } else {
                    eltwises.push_back(conv);
                }
            }

            concats.push_back(eltwises.back());
        }
        rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, mlir::ValueRange(concats),
                                                  isDilatedX ? Dims4D::Act::W : Dims4D::Act::H);
        return mlir::success();
    } else {
        _log.trace("[{0}] expand dilated conv '{1}'", getDebugName(), origOp->getLoc());
        auto dilatedFilter =
                rewriter.create<IE::ExpandDilatedOp>(origOp->getLoc(), origOp.filter(), origOp.dilations());
        rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
                origOp, origOp.input(), dilatedFilter.getResult(), origOp.bias(), origOp.strides(), origOp.pads_begin(),
                origOp.pads_end(), getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})), origOp.post_opAttr());
        return mlir::success();
    }
}

//
// DilatedGroupConvolutionRewriter
//

class DilatedGroupConvolutionRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DilatedGroupConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("DilatedGroupConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DilatedGroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution layer at '{1}'", getDebugName(), origOp->getLoc());

    auto dilatedFilter = rewriter.create<IE::ExpandDilatedOp>(origOp->getLoc(), origOp.filter(), origOp.dilations());
    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, origOp.input(), dilatedFilter.getResult(), origOp.bias(), origOp.strides(), origOp.pads_begin(),
            origOp.pads_end(), getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})), origOp.groupsAttr(),
            origOp.post_opAttr());
    return mlir::success();
}

//
// LegalizeDilatedConvolutionPass
//

class LegalizeDilatedConvolutionPass final : public IE::LegalizeDilatedConvolutionBase<LegalizeDilatedConvolutionPass> {
public:
    explicit LegalizeDilatedConvolutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LegalizeDilatedConvolutionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasSupportedDilations = [](ArrayRef<int64_t> dilations) {
        return dilations[0] == 1 && dilations[1] == 1;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
        return hasSupportedDilations(dilations);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
        return hasSupportedDilations(dilations);
    });
    target.addLegalOp<IE::ExpandDilatedOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::AddOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<DilatedConvolutionRewriter>(&ctx, _log);
    patterns.insert<DilatedGroupConvolutionRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLegalizeDilatedConvolutionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeDilatedConvolutionPass(Logger log) {
    return std::make_unique<LegalizeDilatedConvolutionPass>(log);
}
