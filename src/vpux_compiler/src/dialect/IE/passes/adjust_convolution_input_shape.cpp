//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

// TODO: needs find suitable implict reshape value. Ticket: E#78751
constexpr int64_t CONVOLUTION_INPUT_SHAPE_ALIGNMENT = 4;

//
// ReshapeConvInput
//

class ReshapeConvInput final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ReshapeConvInput(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReshapeConvInput::matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    /*
        Convert 1x1 convolution from
            input          filter                    input               filter
        [1, C, H, 1]     [OC, C, 1, 1]             [1, C, H, 1]        [OC, C, 1, 1]
              \             /                =>        |                   |
                   Conv                            AffineReshape           |
               [1, OC, H, 1]                     [1, C, H/4, 4]            |
                                                       \                  /
                                                              Conv
                                                        [1, OC, H/4, 4]
                                                               |
                                                          AffineReshape
                                                          [1, OC, H, 1]
    */
    auto ctx = convOp->getContext();
    const auto inputShape = getShape(convOp.input());
    const auto filterShape = getShape(convOp.filter());

    // Current logic only works with input and filter shape with 4 dimensions
    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    // check suitable 1x1 convolution with input width = 1, strides = [1, 1]
    if (inputShape[Dims4D::Act::W] != 1 || filterShape[Dims4D::Filter::KX] != 1 ||
        filterShape[Dims4D::Filter::KY] != 1) {
        return mlir::failure();
    }

    const auto strides = parseIntArrayAttr<int64_t>(convOp.strides());
    auto stridesEqualToOne = llvm::all_of(strides, [](const int64_t elem) {
        return elem == 1;
    });
    if (!stridesEqualToOne) {
        return mlir::failure();
    }

    if (inputShape[Dims4D::Act::H] % CONVOLUTION_INPUT_SHAPE_ALIGNMENT != 0) {
        return mlir::failure();
    }

    _log.trace("Adjust input shape for convolution at '{0}'", convOp->getLoc());
    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                             inputShape[Dims4D::Act::H] / CONVOLUTION_INPUT_SHAPE_ALIGNMENT,
                                             CONVOLUTION_INPUT_SHAPE_ALIGNMENT};

    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind()},
                                                   {Dims4D::Act::C.ind()},
                                                   {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()},
                                                   {Dims4D::Act::W.ind()}};
    auto newInput = rewriter.create<IE::AffineReshapeOp>(convOp->getLoc(), convOp.input(),
                                                         getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);
    mlir::BlockAndValueMapping mapper;
    mapper.map(convOp.input(), newInput.output());
    auto newConvOp = mlir::dyn_cast<IE::ConvolutionOp>(rewriter.clone(*convOp, mapper));

    auto outputShape = getShape(convOp.output());
    auto newOutputShape = Shape(SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C],
                                                     outputShape[Dims4D::Act::H] / CONVOLUTION_INPUT_SHAPE_ALIGNMENT,
                                                     outputShape[Dims4D::Act::W] * CONVOLUTION_INPUT_SHAPE_ALIGNMENT});
    auto newOutput = newConvOp.output();
    auto newOutputType = newOutput.getType().dyn_cast<vpux::NDTypeInterface>();
    newOutputType = newOutputType.changeShape(newOutputShape);
    newConvOp.output().setType(newOutputType);
    const auto outShape = getShape(convOp.output()).raw();
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);

    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()},
                                                    {Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::H.ind(), Dims4D::Act::W.ind()}};
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(convOp, newConvOp.output(), getIntArrayOfArray(ctx, outDimMapping),
                                                     outShapeAttr);

    return mlir::success();
}

//
// AdjustConvolutionInputShape
//

class AdjustConvolutionInputShapePass final :
        public IE::AdjustConvolutionInputShapeBase<AdjustConvolutionInputShapePass> {
public:
    explicit AdjustConvolutionInputShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustConvolutionInputShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReshapeConvInput>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertFCToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustConvolutionInputShapePass(Logger log) {
    return std::make_unique<AdjustConvolutionInputShapePass>(log);
}
