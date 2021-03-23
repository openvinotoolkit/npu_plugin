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
    auto inputShape = origOp.input().getType().cast<mlir::ShapedType>().getShape();
    const auto inputShapeType = mlir::RankedTensorType::get({4}, getSInt64Type(origOp->getContext()));

    std::array<int64_t, 4> newInShape = {inputShape[0], inputShape[1], 1, 1};
    const auto inputShapeAttr = mlir::DenseElementsAttr::get(inputShapeType, makeArrayRef(newInShape));
    auto inputShapeOp = rewriter.create<IE::ConstantOp>(origOp->getLoc(), inputShapeType, inputShapeAttr);
    auto newInput = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.input(), inputShapeOp, false);

    auto weightsShape = origOp.weights().getType().cast<mlir::ShapedType>().getShape();
    const auto filterShapeType = mlir::RankedTensorType::get({4}, getSInt64Type(origOp->getContext()));
    std::array<int64_t, 4> newWeightsShape = {weightsShape[0], weightsShape[1], 1, 1};
    const auto filterShapeAttr = mlir::DenseElementsAttr::get(filterShapeType, makeArrayRef(newWeightsShape));
    auto filterShapeOp = rewriter.create<IE::ConstantOp>(origOp->getLoc(), filterShapeType, filterShapeAttr);
    auto newFilter = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.weights(), filterShapeOp, false);

    mlir::Value newBias;
    if (origOp.bias() != nullptr) {
        auto biasShape = origOp.bias().getType().cast<mlir::ShapedType>().getShape();
        const auto biasShapeType = mlir::RankedTensorType::get({4}, getSInt64Type(origOp->getContext()));
        std::array<int64_t, 4> newBiasShape = {biasShape[0], biasShape[1], 1, 1};
        const auto biasShapeAttr = mlir::DenseElementsAttr::get(biasShapeType, makeArrayRef(newBiasShape));
        auto biasShapeOp = rewriter.create<IE::ConstantOp>(origOp->getLoc(), biasShapeType, biasShapeAttr);
        newBias = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.bias(), biasShapeOp, false);
    }

    auto newStrides = getInt32ArrayAttr(origOp->getContext(), ngraph::Strides{1, 1});
    auto newPadsBegin = getInt32ArrayAttr(origOp->getContext(), ngraph::CoordinateDiff{0, 0});
    auto newPadsEnd = getInt32ArrayAttr(origOp->getContext(), ngraph::CoordinateDiff{0, 0});
    auto newDilations = getInt32ArrayAttr(origOp->getContext(), ngraph::Strides{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                     newPadsBegin, newPadsEnd, newDilations);

    auto convShape = convOp.output().getType().cast<mlir::ShapedType>().getShape();
    const auto convShapeType = mlir::RankedTensorType::get({2}, getSInt64Type(origOp->getContext()));
    std::array<int64_t, 2> outputShape = {convShape[0], convShape[1]};
    const auto outputShapeAttr = mlir::DenseElementsAttr::get(convShapeType, makeArrayRef(outputShape));
    auto outputShapeOp = rewriter.create<IE::ConstantOp>(origOp->getLoc(), convShapeType, outputShapeAttr);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, convOp.output(), outputShapeOp, false);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertFCToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::FullyConnectedOp>();
    target.addLegalOp<IE::ConstantOp>();
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
