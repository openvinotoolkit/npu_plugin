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

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

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

    template <class ConcreteOp>
    void updatePoolOperation(ConcreteOp op);

    template <class ConcreteOp>
    void updateConvOperation(ConcreteOp op);
};

//
// safeRunOnFunc
//

void ConvertPaddingsToFloorModePass::safeRunOnFunc() {
    auto func = getFunction();

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

bool canUseSlice(ShapeRef inShape, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd, ArrayRef<int64_t> strides,
                 ArrayRef<int64_t> kernel) {
    VPUX_THROW_UNLESS(inShape.size() == 4, "Wrong input shape");
    VPUX_THROW_UNLESS(strides.size() == 2 && padsBegin.size() == 2 && padsEnd.size() == 2 && kernel.size() == 2,
                      "Wrong strides/padsBegin/padsEnd/kernel");
    static const auto H = vpux::Dims4D::Act::H;
    static const auto W = vpux::Dims4D::Act::W;

    auto sizeH = (inShape[H] - kernel[0]) % strides[0];
    auto sizeW = (inShape[W] - kernel[1]) % strides[1];

    if (padsBegin[0] == 0 && padsBegin[1] == 0 && padsEnd[0] == 0 && padsEnd[1] == 0 && (sizeH != 0 && sizeW != 0)) {
        return true;
    }
    return false;
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updatePoolOperation(ConcreteOp op) {
    const auto inShape = getShape(op.input());
    const auto outShape = getShape(op.output());

    const auto kernel = parseIntArrayAttr<int64_t>(op.kernel_size());
    const auto strides = parseIntArrayAttr<int64_t>(op.strides());
    auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
    auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
    auto useSlice = canUseSlice(inShape, padsBegin, padsEnd, strides, kernel);

    cvtPaddingsToFloorMode(inShape, outShape, kernel, strides, {1, 1}, padsBegin, padsEnd);
    if ((padsEnd.size() == 2) && (padsEnd[0] < 0 || padsEnd[1] < 0) && useSlice) {
        Shape offsets = Shape(inShape.size(), 0);
        const auto offsetsAttr = getIntArrayAttr(op.getContext(), offsets);

        Shape sizes = inShape.raw();
        static const auto H = vpux::Dims4D::Act::H;
        static const auto W = vpux::Dims4D::Act::W;
        sizes[H] = sizes[H] - 1;
        sizes[W] = sizes[W] - 1;
        const auto sizesAttr = getIntArrayAttr(op.getContext(), sizes);

        mlir::OpBuilder builder(op);

        auto sliceOp = builder.create<IE::SliceOp>(op.getLoc(), op.input(), offsetsAttr, sizesAttr);

        op.setOperand(sliceOp.result());
    }

    padsEnd[0] = std::max<int64_t>(padsEnd[0], 0);
    padsEnd[1] = std::max<int64_t>(padsEnd[1], 0);

    const auto newPadsBeginAttr = getIntArrayAttr(op.getContext(), padsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(op.getContext(), padsEnd);

    op.pads_beginAttr(newPadsBeginAttr);
    op.pads_endAttr(newPadsEndAttr);
    op.rounding_typeAttr(IE::RoundingTypeAttr::get(op.getContext(), IE::RoundingType::FLOOR));
}

template <class ConcreteOp>
void ConvertPaddingsToFloorModePass::updateConvOperation(ConcreteOp op) {
    static const auto KY = Dim(2);
    static const auto KX = Dim(3);

    const auto inShape = getShape(op.input());
    const auto filterShape = getShape(op.filter());
    const auto outShape = getShape(op.output());

    const std::array<int64_t, 2> kernel = {filterShape[KY], filterShape[KX]};
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
