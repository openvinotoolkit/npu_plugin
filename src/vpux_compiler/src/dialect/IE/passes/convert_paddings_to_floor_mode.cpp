//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <array>

using namespace vpux;

namespace {

// Normaly the kernel size is smaller than input size with 2 dimensions:
// For 4d NCHW input kernel is 2d XY
// For 5d NCDHW input kernel is 3d XYZ
const size_t INPUT_AND_KERNEL_SIZE_DIFF = 2;

//
// ConvertPaddingsToFloorModePass
//

class ConvertPaddingsToFloorModePass final : public IE::ConvertPaddingsToFloorModeBase<ConvertPaddingsToFloorModePass> {
public:
    explicit ConvertPaddingsToFloorModePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    template <class ConcreteOp>
    void updatePoolOperation(ConcreteOp op);

    template <class ConcreteOp>
    void updateConvOperation(ConcreteOp op);
};

//
// safeRunOnFunc
//

void ConvertPaddingsToFloorModePass::safeRunOnFunc() {
    auto func = getOperation();

    const auto callback = [this](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::MaxPoolOp>([this](IE::MaxPoolOp op) {
                    updatePoolOperation<IE::MaxPoolOp>(op);
                })
                .Case<IE::AvgPoolOp>([this](IE::AvgPoolOp op) {
                    updatePoolOperation<IE::AvgPoolOp>(op);
                })
                .Case<IE::ConvolutionOp>([this](IE::ConvolutionOp op) {
                    updateConvOperation<IE::ConvolutionOp>(op);
                })
                .Case<IE::GroupConvolutionOp>([this](IE::GroupConvolutionOp op) {
                    updateConvOperation<IE::GroupConvolutionOp>(op);
                });
    };

    func.walk(callback);
}

//
// updateOperation
//

int64_t inferInputSizeFloor(int64_t outputSize, int64_t padBegin, int64_t padEnd, int64_t kernel, int64_t stride) {
    return ((outputSize - 1) * stride) - padBegin - padEnd + kernel;
}

void cvtPaddingsToFloorMode(int64_t inputSize, int64_t outputSize, int64_t kernel, int64_t stride, int64_t& padBegin,
                            int64_t& padEnd) {
    const auto inputSizeFloor = inferInputSizeFloor(outputSize, padBegin, padEnd, kernel, stride);

    padEnd = padEnd + (inputSizeFloor - inputSize);
}

void cvtPaddingsToFloorMode(ShapeRef inShape, ShapeRef outShape, ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
                            ArrayRef<int64_t> dilations, MutableArrayRef<int64_t> padsBegin,
                            MutableArrayRef<int64_t> padsEnd) {
    const size_t ksdpSize = inShape.size() - INPUT_AND_KERNEL_SIZE_DIFF;
    VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Wrong input/output shape");
    VPUX_THROW_UNLESS(kernel.size() == ksdpSize && strides.size() == ksdpSize && dilations.size() == ksdpSize &&
                              padsBegin.size() == ksdpSize && padsEnd.size() == ksdpSize && ksdpSize >= 2,
                      "Wrong kernel/strides/dilations/padsBegin/padsEnd");

    for (size_t idx = 0; idx < ksdpSize; idx++) {
        auto inputDim = Dim(idx + INPUT_AND_KERNEL_SIZE_DIFF);

        auto kernelAxis = kernel[idx];

        auto strideAxis = strides[idx];

        auto dilationAxis = dilations[idx];

        auto& padBeginAxis = padsBegin[idx];

        auto& padEndAxis = padsEnd[idx];

        cvtPaddingsToFloorMode(inShape[inputDim], outShape[inputDim], kernelAxis * dilationAxis - (dilationAxis - 1),
                               strideAxis, padBeginAxis, padEndAxis);
    }
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updatePoolOperation(ConcreteOp op) {
    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    const auto kernel = parseIntArrayAttr<int64_t>(op.kernel_size());
    const auto strides = parseIntArrayAttr<int64_t>(op.strides());
    const SmallVector<int64_t> dilatation(kernel.size(), 1);
    auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, dilatation, padsBegin, padsEnd);
    for (size_t idx = 0; idx < padsEnd.size(); idx++) {
        padsEnd[idx] = std::max<int64_t>(padsEnd[idx], 0);
    }

    const auto newPadsBeginAttr = getIntArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(op.getContext(), padsEnd);

    op.pads_beginAttr(newPadsBeginAttr);
    op.pads_endAttr(newPadsEndAttr);
    op.rounding_typeAttr(IE::RoundingTypeAttr::get(op.getContext(), IE::RoundingType::FLOOR));
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updateConvOperation(ConcreteOp op) {
    const auto inShape = getShape(op.input());
    const auto filterShape = getShape(op.filter());
    const auto outShape = getShape(op.output());

    SmallVector<int64_t> kernel;

    for (size_t idx = INPUT_AND_KERNEL_SIZE_DIFF; idx < filterShape.size(); idx++) {
        auto kernelAxis = Dim(idx);
        kernel.push_back(filterShape[kernelAxis]);
    }

    const auto strides = parseIntArrayAttr<int64_t>(op.strides());
    const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());
    auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, dilations, padsBegin, padsEnd);
    for (size_t i = 0; i < padsEnd.size(); ++i) {
        padsEnd[i] = std::max<int64_t>(padsEnd[i], 0);
    }

    const auto newPadsBeginAttr = getIntArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(op.getContext(), padsEnd);

    op.pads_beginAttr(newPadsBeginAttr);
    op.pads_endAttr(newPadsEndAttr);
}

}  // namespace

//
// createConvertPaddingsToFloorModePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPaddingsToFloorModePass(Logger log) {
    return std::make_unique<ConvertPaddingsToFloorModePass>(log);
}
