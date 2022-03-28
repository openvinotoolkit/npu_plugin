//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph/coordinate_diff.hpp>
#include <ngraph/op/op.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertFCToConvPass
//

class ConvertFCToConvPass final : public IE::ConvertFCToConvBase<ConvertFCToConvPass> {
public:
    explicit ConvertFCToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class FullyConnectedOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// FullyConnectedOpConverter
//

class ConvertFCToConvPass::FullyConnectedOpConverter final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    FullyConnectedOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertFCToConvPass::FullyConnectedOpConverter::matchAndRewrite(
        IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = origOp.input().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 4> newInShape = {inputShape[0], inputShape[1], 1, 1};
    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    auto newInput = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, inputShapeAttr);

    const auto weightsShape = origOp.weights().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 4> newWeightsShape = {weightsShape[0], weightsShape[1], 1, 1};
    const auto filterShapeAttr = getIntArrayAttr(getContext(), newWeightsShape);
    auto newFilter =
            rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.weights(), nullptr, false, filterShapeAttr);

    mlir::Value newBias;
    if (origOp.bias() != nullptr) {
        const auto biasShape = origOp.bias().getType().cast<vpux::NDTypeInterface>().getShape().raw();
        const std::array<int64_t, 4> newBiasShape = {biasShape[0], biasShape[1], 1, 1};
        const auto biasShapeAttr = getIntArrayAttr(getContext(), newBiasShape);
        newBias = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.bias(), nullptr, false, biasShapeAttr);
    }

    auto newStrides = getIntArrayAttr(getContext(), ngraph::Strides{1, 1});
    auto newPadsBegin = getIntArrayAttr(getContext(), ngraph::CoordinateDiff{0, 0});
    auto newPadsEnd = getIntArrayAttr(getContext(), ngraph::CoordinateDiff{0, 0});
    auto newDilations = getIntArrayAttr(getContext(), ngraph::Strides{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                     newPadsBegin, newPadsEnd, newDilations, nullptr);

    const auto convShape = convOp.output().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 2> outputShape = {convShape[0], convShape[1]};
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, convOp.output(), nullptr, false, outputShapeAttr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertFCToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::FullyConnectedOp>();
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FullyConnectedOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertFCToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertFCToConvPass(Logger log) {
    return std::make_unique<ConvertFCToConvPass>(log);
}
