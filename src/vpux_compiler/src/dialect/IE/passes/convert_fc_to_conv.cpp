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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <ngraph_ops/convolution_ie.hpp>

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
    const auto inputShape = origOp.input().getType().cast<mlir::ShapedType>().getShape();
    const std::array<int64_t, 4> newInShape = {inputShape[0], inputShape[1], 1, 1};
    const auto inputShapeAttr = getInt64ArrayAttr(getContext(), newInShape);
    auto newInput = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), nullptr, false, inputShapeAttr);

    const auto weightsShape = origOp.weights().getType().cast<mlir::ShapedType>().getShape();
    const std::array<int64_t, 4> newWeightsShape = {weightsShape[0], weightsShape[1], 1, 1};
    const auto filterShapeAttr = getInt64ArrayAttr(getContext(), newWeightsShape);
    auto newFilter =
            rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.weights(), nullptr, false, filterShapeAttr);

    mlir::Value newBias;
    if (origOp.bias() != nullptr) {
        const auto biasShape = origOp.bias().getType().cast<mlir::ShapedType>().getShape();
        const std::array<int64_t, 4> newBiasShape = {biasShape[0], biasShape[1], 1, 1};
        const auto biasShapeAttr = getInt64ArrayAttr(getContext(), newBiasShape);
        newBias = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.bias(), nullptr, false, biasShapeAttr);
    }

    auto newStrides = getInt32ArrayAttr(getContext(), ngraph::Strides{1, 1});
    auto newPadsBegin = getInt32ArrayAttr(getContext(), ngraph::CoordinateDiff{0, 0});
    auto newPadsEnd = getInt32ArrayAttr(getContext(), ngraph::CoordinateDiff{0, 0});
    auto newDilations = getInt32ArrayAttr(getContext(), ngraph::Strides{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                     newPadsBegin, newPadsEnd, newDilations, nullptr);

    const auto convShape = convOp.output().getType().cast<mlir::ShapedType>().getShape();
    const std::array<int64_t, 2> outputShape = {convShape[0], convShape[1]};
    const auto outputShapeAttr = getInt64ArrayAttr(getContext(), outputShape);
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
