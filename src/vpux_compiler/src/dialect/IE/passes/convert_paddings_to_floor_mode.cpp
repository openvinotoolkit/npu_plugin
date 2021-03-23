//
// Copyright Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <array>

using namespace vpux;

namespace {

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

    void updateOperation(IE::MaxPoolOp op);
    void updateOperation(IE::AvgPoolOp op);
    void updateOperation(IE::ConvolutionOp op);
};

//
// safeRunOnFunc
//

void ConvertPaddingsToFloorModePass::safeRunOnFunc() {
    auto func = getFunction();

    const auto callback = [this](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::MaxPoolOp>([this](IE::MaxPoolOp op) {
                    updateOperation(op);
                })
                .Case<IE::AvgPoolOp>([this](IE::AvgPoolOp op) {
                    updateOperation(op);
                })
                .Case<IE::ConvolutionOp>([this](IE::ConvolutionOp op) {
                    updateOperation(op);
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
    padEnd = std::max<int64_t>(padEnd, 0);
}

void cvtPaddingsToFloorMode(ShapeRef inShape, ShapeRef outShape, ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
                            ArrayRef<int64_t> dilations, MutableArrayRef<int64_t> padsBegin,
                            MutableArrayRef<int64_t> padsEnd) {
    static const auto H = Dim(2);
    static const auto W = Dim(3);

    VPUX_THROW_UNLESS(inShape.size() == 4 && outShape.size() == 4, "Wrong input/output shape");
    VPUX_THROW_UNLESS(kernel.size() == 2 && strides.size() == 2 && dilations.size() == 2 && padsBegin.size() == 2 &&
                              padsEnd.size() == 2,
                      "Wrong kernel/strides/dilations/padsBegin/padsEnd");

    const auto kernelY = kernel[0];
    const auto kernelX = kernel[1];

    const auto strideY = strides[0];
    const auto strideX = strides[1];

    const auto dilationY = dilations[0];
    const auto dilationX = dilations[1];

    auto& padBeginY = padsBegin[0];
    auto& padBeginX = padsBegin[1];

    auto& padEndY = padsEnd[0];
    auto& padEndX = padsEnd[1];

    cvtPaddingsToFloorMode(inShape[H], outShape[H], kernelY * dilationY - (dilationY - 1), strideY, padBeginY, padEndY);
    cvtPaddingsToFloorMode(inShape[W], outShape[W], kernelX * dilationX - (dilationX - 1), strideX, padBeginX, padEndX);
}

void ConvertPaddingsToFloorModePass::updateOperation(IE::MaxPoolOp op) {
    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    const auto kernel = parseIntArrayAttr(op.kernel_size());
    const auto strides = parseIntArrayAttr(op.strides());
    auto padsBegin = parseIntArrayAttr(op.pads_begin());
    auto padsEnd = parseIntArrayAttr(op.pads_end());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, {1, 1}, padsBegin, padsEnd);

    const auto newPadsBeginAttr = getInt32ArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getInt32ArrayAttr(op.getContext(), padsEnd);

    op.pads_beginAttr(newPadsBeginAttr);
    op.pads_endAttr(newPadsEndAttr);
    op.rounding_typeAttr(IE::RoundingTypeAttr::get(op.getContext(), IE::RoundingType::FLOOR));
}

void ConvertPaddingsToFloorModePass::updateOperation(IE::AvgPoolOp op) {
    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    const auto kernel = parseIntArrayAttr(op.kernel_size());
    const auto strides = parseIntArrayAttr(op.strides());
    auto padsBegin = parseIntArrayAttr(op.pads_begin());
    auto padsEnd = parseIntArrayAttr(op.pads_end());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, {1, 1}, padsBegin, padsEnd);

    const auto newPadsBeginAttr = getInt32ArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getInt32ArrayAttr(op.getContext(), padsEnd);

    op.pads_beginAttr(newPadsBeginAttr);
    op.pads_endAttr(newPadsEndAttr);
    op.rounding_typeAttr(IE::RoundingTypeAttr::get(op.getContext(), IE::RoundingType::FLOOR));
}

void ConvertPaddingsToFloorModePass::updateOperation(IE::ConvolutionOp op) {
    static const auto KY = Dim(2);
    static const auto KX = Dim(3);

    const auto inShape = getShape(op.input());
    const auto filterShape = getShape(op.filter());
    const auto outShape = getShape(op.output());

    const std::array<int64_t, 2> kernel = {filterShape[KY], filterShape[KX]};
    const auto strides = parseIntArrayAttr(op.strides());
    const auto dilations = parseIntArrayAttr(op.dilations());
    auto padsBegin = parseIntArrayAttr(op.pads_begin());
    auto padsEnd = parseIntArrayAttr(op.pads_end());

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, dilations, padsBegin, padsEnd);

    const auto newPadsBeginAttr = getInt32ArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getInt32ArrayAttr(op.getContext(), padsEnd);

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
