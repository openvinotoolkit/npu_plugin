//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
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

static mlir::Value getZerosConst(mlir::PatternRewriter& rewriter, Shape constShape, IE::GroupConvolutionOp origOp) {
    const auto elemType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(constShape), elemType);

    mlir::DenseElementsAttr denseElementVal = wrapData(dataStorageType, 0.0f);
    VPUX_THROW_UNLESS(denseElementVal != nullptr,
                      "GroupConvolutionOp has incompatible data type {0}, only float16 or float32 are supported",
                      elemType);

    return rewriter.create<Const::DeclareOp>(origOp.getLoc(), dataStorageType, Const::ContentAttr::get(denseElementVal))
            .output();
}

// Here is an optimization for the case in the below:
// 1.if the expanded kernel size beyond NCE HW spec
// 2. Y dialation or X dilation is 1
// Assume Y dilation is 1, filter is k*k, the optimization is working as below:
// step 1: Slice filter to k slices with sliceShape k*1(Y*X)
// step 2: Slice activtion to k slices with shape actSliceShape, we consider padding case here for activation slicing.
// step 3: Use filter_slice[n] and activation_slice[n] to do non-dilated group convoulution, generate output tensor
// slice -output[n]
// step 4: Insert Eltwises.Add with each output tensor slice to get the finnal result.

mlir::LogicalResult DilatedGroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.dilations()));
    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.pads_begin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.pads_end()));
    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto expandKernelX = (KX - 1) * dilations[Dims4D::Dilation::X] + 1;
    const auto expandKernelY = (KY - 1) * dilations[Dims4D::Dilation::Y] + 1;

    if ((expandKernelX > vpux::VPU::NCEInvariant::MAX_KERNEL_SIZE && dilations[Dims4D::Dilation::Y] == 1) ||
        (expandKernelY > vpux::VPU::NCEInvariant::MAX_KERNEL_SIZE && dilations[Dims4D::Dilation::X] == 1)) {
        _log.trace("[{0}] Slice Group Dilated conv to small task '{1}'", getDebugName(), origOp->getLoc());

        mlir::MLIRContext* ctx = origOp->getContext();
        const auto inputShape = getShape(origOp->getOperand(0));
        const auto IC = filterShape[Dims4D::Filter::IC];
        const auto OC = filterShape[Dims4D::Filter::OC];
        const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));
        const auto broadcastType =
                vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
        bool isDilatedX = dilations[Dims4D::Dilation::Y] == 1;

        // Filter slices
        SmallVector<mlir::Value> slicedFilters;
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

        int64_t windowW = ((padStart[Dims4D::PadsBegin::Left] + inputShape[Dims4D::Act::W] +
                            padEnd[Dims4D::PadsEnd::Right] - expandKernelX) /
                           strides[Dims4D::Strides::X]) *
                                  strides[Dims4D::Strides::X] +
                          1;
        int64_t windowH = ((padStart[Dims4D::PadsBegin::Top] + inputShape[Dims4D::Act::H] +
                            padEnd[Dims4D::PadsEnd::Bottom] - expandKernelY) /
                           strides[Dims4D::Strides::Y]) *
                                  strides[Dims4D::Strides::Y] +
                          1;

        int64_t validLeftBoundary = 0 - padEnd[Dims4D::PadsBegin::Left];
        int64_t validRightBoundary = inputShape[Dims4D::Act::W] - 1 + padEnd[Dims4D::PadsEnd::Right];
        int64_t validTopBoundary = 0 - padEnd[Dims4D::PadsBegin::Top];
        int64_t validBottomBoundary = inputShape[Dims4D::Act::H] - 1 + padEnd[Dims4D::PadsEnd::Bottom];

        // Activation slices
        SmallVector<mlir::Value> accumulativeOutputTensors;
        for (int64_t k = 0; k < kernel; k++) {
            SmallVector<mlir::Value> concatConsConv;
            Shape newPadStart(padStart.size());
            Shape newPadEnd(padEnd.size());

            newPadStart[Dims4D::PadsBegin::Left] = isDilatedX ? 0 : padStart[Dims4D::PadsBegin::Left];
            newPadEnd[Dims4D::PadsEnd::Right] = isDilatedX ? 0 : padEnd[Dims4D::PadsEnd::Right];
            newPadStart[Dims4D::PadsBegin::Top] = isDilatedX ? padStart[Dims4D::PadsBegin::Top] : 0;
            newPadEnd[Dims4D::PadsEnd::Bottom] = isDilatedX ? padEnd[Dims4D::PadsEnd::Bottom] : 0;

            int64_t startW = isDilatedX ? (k * dilations[Dims4D::Dilation::X] - padStart[Dims4D::PadsBegin::Left]) : 0;
            VPUX_THROW_WHEN(startW > validRightBoundary, "Dimension W of size '{0}' is out of range '[{1}, {2}]''",
                            startW, validLeftBoundary, validRightBoundary);
            int64_t startH = isDilatedX ? 0 : (k * dilations[Dims4D::Dilation::Y] - padStart[Dims4D::PadsBegin::Top]);
            VPUX_THROW_WHEN(startH > validBottomBoundary, "Dimension H of size '{0}' is out of range '[{1}, {2}]'",
                            startH, validTopBoundary, validBottomBoundary);

            int64_t endW = isDilatedX ? (startW + windowW - 1) : (inputShape[Dims4D::Act::W] - 1);
            VPUX_THROW_WHEN(endW > validRightBoundary, "Dimension W of size '{0}' is out of range '[{1}, {2}]''", endW,
                            validLeftBoundary, validRightBoundary);
            int64_t endH = isDilatedX ? (inputShape[Dims4D::Act::H] - 1) : (startH + windowH - 1);
            VPUX_THROW_WHEN(endH > validBottomBoundary, "Dimension H of size '{0}' is out of range '[{1}, {2}]'", endH,
                            validTopBoundary, validBottomBoundary);

            bool isLastConvSlice = false;
            bool isAloneConvSlice = false;

            Shape offsets(inputShape.size());

            SmallVector<int64_t> actSliceShape;

            // Caculate actSliceShape and offset in case of X Dilation
            auto noOverlapWithRange = [&](int64_t start, int64_t end, int64_t rangeStart, int64_t rangeEnd) -> bool {
                return end < rangeStart || start > rangeEnd;
            };

            if (isDilatedX) {
                if (noOverlapWithRange(startW, endW, 0, inputShape[Dims4D::Act::W] - 1)) {
                    _log.trace("[{0}] Skip Activation slice in loc '{1}', startW = {2}, endW = {3}, k = {4}",
                               getDebugName(), origOp->getLoc(), startW, endW, k);
                    continue;
                } else {
                    actSliceShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::H],
                                     std::min(endW, inputShape[Dims4D::Act::W] - 1) -
                                             std::max(static_cast<int64_t>(0), startW) + 1};

                    offsets[Dims4D::Act::W] = std::max(static_cast<int64_t>(0), startW);
                    offsets[Dims4D::Act::H] = startH;

                    if (k == (kernel - 1)) {
                        isLastConvSlice = true;
                    } else {
                        if (noOverlapWithRange(
                                    startW + dilations[Dims4D::Dilation::X], endW + dilations[Dims4D::Dilation::X], 0,
                                    inputShape[Dims4D::Act::W] - 1))  // Next conv input slice is out of window
                            isLastConvSlice = true;
                    }

                    if (isLastConvSlice) {
                        if (k == 0) {
                            isAloneConvSlice = true;
                        } else if (noOverlapWithRange(
                                           startW - dilations[Dims4D::Dilation::X],
                                           endW - dilations[Dims4D::Dilation::X], 0,
                                           inputShape[Dims4D::Act::W] - 1))  // Prev conv input slice is out of window
                            isAloneConvSlice = true;
                    }
                    _log.trace("[{0}] Create Activation slice in loc '{1}', startW = {2}, endW = {3}, k = {4}, "
                               "isLastConvSlice = {5}, isAloneConvSlice = {6}",
                               getDebugName(), origOp->getLoc(), startW, endW, k, isLastConvSlice, isAloneConvSlice);
                }
            }
            // Caculate actSliceShape and offset in case of Y Dilation
            else {
                if (noOverlapWithRange(startH, endH, 0, inputShape[Dims4D::Act::H] - 1)) {
                    _log.trace("[{0}] Skip Activation slice in loc '{1}', startH = {2}, endH = {3}, k = {4}",
                               getDebugName(), origOp->getLoc(), startH, endH, k);
                    continue;
                } else {
                    actSliceShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                     std::min(endH, inputShape[Dims4D::Act::H] - 1) -
                                             std::max(static_cast<int64_t>(0), startH) + 1,
                                     inputShape[Dims4D::Act::W]};

                    offsets[Dims4D::Act::W] = startW;
                    offsets[Dims4D::Act::H] = std::max(static_cast<int64_t>(0), startH);

                    if (k == (kernel - 1)) {
                        isLastConvSlice = true;
                    } else {
                        if (noOverlapWithRange(
                                    startH + dilations[Dims4D::Dilation::Y], endH + dilations[Dims4D::Dilation::Y], 0,
                                    inputShape[Dims4D::Act::H] - 1))  // Next conv input slice is out of window
                            isLastConvSlice = true;
                    }

                    if (isLastConvSlice) {
                        if (k == 0) {
                            isAloneConvSlice = true;
                        } else if (noOverlapWithRange(
                                           startH - dilations[Dims4D::Dilation::Y],
                                           endH - dilations[Dims4D::Dilation::Y], 0,
                                           inputShape[Dims4D::Act::H] - 1))  // Prev conv input slice is out of window
                            isAloneConvSlice = true;
                    }
                    _log.trace("[{0}] Create Activation slice in loc '{1}', startH = {2}, endH = {3}, k = {4}, "
                               "isLastConvSlice = {5}, isAloneConvSlice = {6}",
                               getDebugName(), origOp->getLoc(), startH, endH, k, isLastConvSlice, isAloneConvSlice);
                }
            }

            // Left or up padding
            auto inRange = [&](int64_t val, int64_t rangeStart, int64_t rangeEnd) -> bool {
                return val >= rangeStart && val <= rangeEnd;
            };

            if (!inRange(isDilatedX ? startW : startH, 0,
                         isDilatedX ? (inputShape[Dims4D::Act::W] - 1) : (inputShape[Dims4D::Act::H] - 1))) {
                Shape zeroConstShape(inputShape.size());
                zeroConstShape[Dims4D::Act::N] = inputShape[Dims4D::Act::N];
                zeroConstShape[Dims4D::Act::C] = inputShape[Dims4D::Act::C];
                zeroConstShape[Dims4D::Act::H] = isDilatedX ? inputShape[Dims4D::Act::H] : (0 - startH);
                zeroConstShape[Dims4D::Act::W] = isDilatedX ? (0 - startW) : inputShape[Dims4D::Act::W];

                auto constZeros = getZerosConst(rewriter, zeroConstShape, origOp);
                concatConsConv.push_back(constZeros);
            }

            // Slice activation if necessary
            auto coversTheRange = [&](int64_t start, int64_t end, int64_t rangeStart, int64_t rangeEnd) -> bool {
                return start <= rangeStart && end >= rangeEnd;
            };

            if (isDilatedX ? coversTheRange(startW, endW, 0, inputShape[Dims4D::Act::W] - 1)
                           : coversTheRange(startH, endH, 0, inputShape[Dims4D::Act::H] - 1))
                concatConsConv.push_back(origOp.input());
            else {
                mlir::Value convInput;
                convInput = rewriter.create<IE::SliceOp>(origOp->getLoc(), origOp.input(),
                                                         getIntArrayAttr(ctx, offsets.raw()),
                                                         getIntArrayAttr(ctx, actSliceShape));

                concatConsConv.push_back(convInput);
            }

            // Right or bottom padding
            if (!inRange(isDilatedX ? endW : endH, 0,
                         isDilatedX ? (inputShape[Dims4D::Act::W] - 1) : (inputShape[Dims4D::Act::H] - 1))) {
                Shape zeroConstShape(inputShape.size());
                zeroConstShape[Dims4D::Act::N] = inputShape[Dims4D::Act::N];
                zeroConstShape[Dims4D::Act::C] = inputShape[Dims4D::Act::C];
                zeroConstShape[Dims4D::Act::H] =
                        isDilatedX ? inputShape[Dims4D::Act::H] : (endH - inputShape[Dims4D::Act::H] + 1);
                zeroConstShape[Dims4D::Act::W] =
                        isDilatedX ? (endW - inputShape[Dims4D::Act::W] + 1) : inputShape[Dims4D::Act::W];

                auto constZeros = getZerosConst(rewriter, zeroConstShape, origOp);
                concatConsConv.push_back(constZeros);
            }

            // Padded group conv input
            auto concat = rewriter.create<IE::ConcatOp>(origOp.getLoc(), mlir::ValueRange(concatConsConv),
                                                        isDilatedX ? Dims4D::Act::W : Dims4D::Act::H);

            // add bias and post process for the last convolution and eltwise.
            auto conv = rewriter.create<IE::GroupConvolutionOp>(
                    origOp->getLoc(), concat, slicedFilters[k],
                    (isLastConvSlice) ? origOp.bias() : nullptr,  // Add bias for the last conv slice
                    origOp.strides(), getIntArrayAttr(origOp->getContext(), newPadStart),
                    getIntArrayAttr(origOp->getContext(), newPadEnd),
                    getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})), origOp.groupsAttr(),
                    (isAloneConvSlice)
                            ? origOp.post_opAttr()
                            : nullptr);  // Add postOp if this is the only one conv slice(No AddOp in this case)

            if (accumulativeOutputTensors.size() > 0) {
                VPUX_THROW_WHEN(isAloneConvSlice, "Conflict with isAloneConvSlice flag");
                auto add = rewriter.create<IE::AddOp>(
                        origOp->getLoc(), accumulativeOutputTensors.back(), conv, broadcastType,
                        (isLastConvSlice)
                                ? origOp.post_opAttr()
                                : nullptr);  // Not only 1 conv, we have AddOp here and add the postOp to AddOp
                accumulativeOutputTensors.push_back(add);
            } else {
                accumulativeOutputTensors.push_back(conv);
            }
        }
        rewriter.replaceOp(origOp, accumulativeOutputTensors.back());

        return mlir::success();
    } else {
        _log.trace("[{0}] expand dilated conv '{1}'", getDebugName(), origOp->getLoc());
        auto dilatedFilter =
                rewriter.create<IE::ExpandDilatedOp>(origOp->getLoc(), origOp.filter(), origOp.dilations());
        rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
                origOp, origOp.input(), dilatedFilter.getResult(), origOp.bias(), origOp.strides(), origOp.pads_begin(),
                origOp.pads_end(), getIntArrayAttr(origOp->getContext(), makeArrayRef({1, 1})), origOp.groupsAttr(),
                origOp.post_opAttr());
        return mlir::success();
    }
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
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DilatedConvolutionRewriter>(&ctx, _log);
    patterns.add<DilatedGroupConvolutionRewriter>(&ctx, _log);

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
