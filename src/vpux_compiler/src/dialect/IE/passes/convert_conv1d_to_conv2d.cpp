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
// ConvertConv1DToConv2DPass
//

class ConvertConv1DToConv2DPass final : public IE::ConvertConv1DToConv2DBase<ConvertConv1DToConv2DPass> {
public:
    explicit ConvertConv1DToConv2DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    static mlir::Value extendTensor(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input);
    static mlir::ArrayAttr append(mlir::MLIRContext* context, mlir::ArrayAttr attr, int64_t value);

public:
    class ConvolutionExpansion;
    class GroupConvolutionExpansion;

private:
    void safeRunOnFunc() final;
};

mlir::Value ConvertConv1DToConv2DPass::extendTensor(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                    mlir::Value input) {
    if (!input) {
        return {};
    }

    // Extend shape with Height = 1. [N C W] -> [N C 1 W]
    const auto shape = input.getType().cast<mlir::ShapedType>().getShape();
    auto newShape = to_small_vector(shape);
    newShape.insert(newShape.end() - 1, 1);

    const auto newShapeAttr = getInt64ArrayAttr(rewriter.getContext(), newShape);
    return rewriter.create<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
}

mlir::ArrayAttr ConvertConv1DToConv2DPass::append(mlir::MLIRContext* context, mlir::ArrayAttr attr, int64_t value) {
    auto vector = parseIntArrayAttr(attr);
    vector.insert(vector.begin(), value);
    return getInt32ArrayAttr(context, vector);
}

//
// ConvolutionExpansion
//

class ConvertConv1DToConv2DPass::ConvolutionExpansion final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionExpansion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertConv1DToConv2DPass::ConvolutionExpansion::matchAndRewrite(
        IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::Convolution Operation '{0}'", origOp->getLoc());

    auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp.input());
    auto newFilter = extendTensor(rewriter, origOp->getLoc(), origOp.filter());
    auto newBias = extendTensor(rewriter, origOp->getLoc(), origOp.bias());

    const auto newStrides = append(getContext(), origOp.strides(), 1);
    const auto newPadsBegin = append(getContext(), origOp.pads_begin(), 0);
    const auto newPadsEnd = append(getContext(), origOp.pads_end(), 0);
    const auto newDilations = append(getContext(), origOp.dilations(), 1);

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                        newPadsBegin, newPadsEnd, newDilations, origOp.post_opAttr());

    const auto outputShape = origOp.output().getType().cast<mlir::ShapedType>().getShape();
    const auto outputShapeAttr = getInt64ArrayAttr(getContext(), outputShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 'IE::Convolution' (2D)");

    return mlir::success();
}

//
// GroupConvolutionExpansion
//

class ConvertConv1DToConv2DPass::GroupConvolutionExpansion final :
        public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvolutionExpansion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertConv1DToConv2DPass::GroupConvolutionExpansion::matchAndRewrite(
        IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::GroupConvolution Operation '{0}'", origOp->getLoc());

    auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp.input());
    auto newFilter = extendTensor(rewriter, origOp->getLoc(), origOp.filter());
    auto newBias = extendTensor(rewriter, origOp->getLoc(), origOp.bias());

    const auto newStrides = append(getContext(), origOp.strides(), 1);
    const auto newPadsBegin = append(getContext(), origOp.pads_begin(), 0);
    const auto newPadsEnd = append(getContext(), origOp.pads_end(), 0);
    const auto newDilations = append(getContext(), origOp.dilations(), 1);

    auto newConvOp =
            rewriter.create<IE::GroupConvolutionOp>(origOp->getLoc(), newInput, newFilter, newBias, newStrides,
                                                    newPadsBegin, newPadsEnd, newDilations, origOp.groupsAttr());

    const auto outputShape = origOp.output().getType().cast<mlir::ShapedType>().getShape();
    const auto outputShapeAttr = getInt64ArrayAttr(getContext(), outputShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp.output(), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 'IE::GroupConvolution' (2D)");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertConv1DToConv2DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalConvOp = [&](IE::ConvolutionOp conv) {
        const auto inputShape = conv.input().getType().cast<mlir::ShapedType>().getShape();
        return inputShape.size() != 3;
    };

    const auto isLegalGroupConvOp = [&](IE::GroupConvolutionOp groupConv) {
        const auto inputShape = groupConv.filter().getType().cast<mlir::ShapedType>().getShape();
        const auto hasGroups = groupConv.groups().getValueOr(0) != 0;
        return (inputShape.size() + hasGroups) != 4;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalConvOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalGroupConvOp);
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvolutionExpansion>(&ctx, _log);
    patterns.insert<GroupConvolutionExpansion>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConv1DToConv2DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConv1DToConv2DPass(Logger log) {
    return std::make_unique<ConvertConv1DToConv2DPass>(log);
}
