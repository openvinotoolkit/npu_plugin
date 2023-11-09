//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

mlir::Value extendTensor(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input) {
    if (!input) {
        return nullptr;
    }

    // Extend shape with Height = 1. [N C W] -> [N C 1 W]
    const auto shape = input.getType().cast<vpux::NDTypeInterface>().getShape();

    auto newShape = to_small_vector(shape);
    newShape.insert(newShape.end() - 1, 1);

    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    return rewriter.createOrFold<IE::ReshapeOp>(loc, input, nullptr, false, newShapeAttr);
}

mlir::ArrayAttr append(mlir::MLIRContext* context, mlir::ArrayAttr attr, int64_t value) {
    auto vector = parseIntArrayAttr<int64_t>(attr);
    vector.insert(vector.begin(), value);
    return getIntArrayAttr(context, vector);
}

//
// ConvGeneralExpansion
//
template <class ConcreteOp>
class ConvGeneralExpansion final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvGeneralExpansion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ConvGeneralExpansion<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("Convert NCE to 4D for '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto* ctx = origOp->getContext();

    const auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp->getOperand(0));
    const auto newFilter = extendTensor(rewriter, origOp->getLoc(), origOp.filter());

    const auto newStrides = append(ctx, origOp.strides(), 1);
    const auto newPadsBegin = append(ctx, origOp.pads_begin(), 0);
    const auto newPadsEnd = append(ctx, origOp.pads_end(), 0);
    const auto newDilations = append(ctx, origOp.dilations(), 1);

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{newInput, newFilter});
    auto* newConvOp = rewriter.clone(*origOp.getOperation(), mapper);

    VPUX_THROW_UNLESS(newConvOp->hasAttr("pads_begin") && newConvOp->hasAttr("pads_end") &&
                              newConvOp->hasAttr("strides") && newConvOp->hasAttr("dilations"),
                      "Cannot get all attributions");
    newConvOp->setAttr("pads_begin", newPadsBegin);
    newConvOp->setAttr("pads_end", newPadsEnd);
    newConvOp->setAttr("strides", newStrides);
    newConvOp->setAttr("dilations", newDilations);

    if (auto deConv = mlir::dyn_cast<IE::DeconvolutionOp>(origOp.getOperation())) {
        const auto newOutputPadding = append(ctx, deConv.output_padding(), 0);
        newConvOp->setAttr("output_padding", newOutputPadding);
    }

    vpux::inferReturnTypes(newConvOp, vpux::InferShapedTypeMode::ALL);

    const auto outputType = origOp.output().getType().template dyn_cast<vpux::NDTypeInterface>();
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputType.getShape());
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConvOp->getResult(0), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 4D '{0}'", origOp->getName());
    return mlir::success();
}

//
// PoolingGeneralExpansion
//
template <class ConcreteOp>
class PoolingGeneralExpansion final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    PoolingGeneralExpansion(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult PoolingGeneralExpansion<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Convert NCE to 4D for '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto* ctx = origOp->getContext();

    const auto newInput = extendTensor(rewriter, origOp->getLoc(), origOp.input());
    const auto newKernelSize = append(ctx, origOp.kernel_size(), 1);
    const auto newStrides = append(ctx, origOp.strides(), 1);
    const auto newPadsBegin = append(ctx, origOp.pads_begin(), 0);
    const auto newPadsEnd = append(ctx, origOp.pads_end(), 0);

    mlir::BlockAndValueMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{newInput});
    auto* newPoolingOp = rewriter.clone(*origOp.getOperation(), mapper);

    VPUX_THROW_UNLESS(newPoolingOp->hasAttr("pads_begin") && newPoolingOp->hasAttr("pads_end") &&
                              newPoolingOp->hasAttr("strides") && newPoolingOp->hasAttr("kernel_size"),
                      "Cannot get all attributions");
    newPoolingOp->setAttr("pads_begin", newPadsBegin);
    newPoolingOp->setAttr("pads_end", newPadsEnd);
    newPoolingOp->setAttr("strides", newStrides);
    newPoolingOp->setAttr("kernel_size", newKernelSize);
    vpux::inferReturnTypes(newPoolingOp, vpux::InferShapedTypeMode::ALL);

    const auto outputType = origOp.output().getType().template dyn_cast<vpux::NDTypeInterface>();
    const auto outputShapeAttr = getIntArrayAttr(ctx, outputType.getShape());
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newPoolingOp->getResult(0), nullptr, false, outputShapeAttr);

    _log.trace("Replaced with 4D '{0}'", origOp->getName());
    return mlir::success();
}

//
// ConvertNceOpsTo4DPass
//

class ConvertNceOpsTo4DPass final : public IE::ConvertNceOpsTo4DBase<ConvertNceOpsTo4DPass> {
public:
    explicit ConvertNceOpsTo4DPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertNceOpsTo4DPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalNceOp = [&](mlir::Operation* op) {
        const auto inputShape = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getShape();
        return inputShape.size() != 3;
    };

    const auto isLegalGroupConvOp = [&](IE::GroupConvolutionOp groupConv) {
        const auto inputShape = groupConv.filter().getType().cast<vpux::NDTypeInterface>().getShape();
        const auto hasGroups = groupConv.groups().value_or(0) != 0;
        return (inputShape.size() + hasGroups) != 4;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(isLegalNceOp);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalGroupConvOp);
    target.addDynamicallyLegalOp<IE::DeconvolutionOp>(isLegalNceOp);
    target.addDynamicallyLegalOp<IE::MaxPoolOp>(isLegalNceOp);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(isLegalNceOp);
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvGeneralExpansion<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralExpansion<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralExpansion<IE::DeconvolutionOp>>(&ctx, _log);
    patterns.add<PoolingGeneralExpansion<IE::MaxPoolOp>>(&ctx, _log);
    patterns.add<PoolingGeneralExpansion<IE::AvgPoolOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertNceOpsTo4DPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNceOpsTo4DPass(Logger log) {
    return std::make_unique<ConvertNceOpsTo4DPass>(log);
}
