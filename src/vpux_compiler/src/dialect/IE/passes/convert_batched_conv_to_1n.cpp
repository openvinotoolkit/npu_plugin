//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <mlir/IR/BlockAndValueMapping.h>

using namespace vpux;

namespace {

//
// LayerConverter
//

class LayerConverter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    LayerConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("LayerConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayerConverter::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Conv: {0}", origOp);
    auto originShape = getShape(origOp.input());

    SmallVector<unsigned> transPerm(originShape.size(), 0);
    transPerm[Dims4D::Act::N.ind()] = checked_cast<unsigned>(Dims4D::Act::H.ind());
    transPerm[Dims4D::Act::C.ind()] = checked_cast<unsigned>(Dims4D::Act::C.ind());
    transPerm[Dims4D::Act::H.ind()] = checked_cast<unsigned>(Dims4D::Act::N.ind());
    transPerm[Dims4D::Act::W.ind()] = checked_cast<unsigned>(Dims4D::Act::W.ind());

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transPerm, origOp->getContext()));
    auto inTransposeLoc = appendLoc(origOp->getLoc(), "_ConvertBatchedConv_inTranspose");
    auto inTranspose = rewriter.create<IE::TransposeOp>(inTransposeLoc, origOp.input(), nullptr, orderAttr);
    _log.trace("Insert transpose for activation: {0}", inTranspose);

    auto newConvLoc = appendLoc(origOp->getLoc(), "_ConvertBatchedConv_newConv");
    auto convOp = rewriter.create<IE::ConvolutionOp>(
            newConvLoc, inTranspose.output(), origOp.filter(), origOp.bias(), origOp.stridesAttr(),
            origOp.pads_beginAttr(), origOp.pads_endAttr(), origOp.dilationsAttr(), origOp.post_opAttr());

    // Support mixed precision convolution i8 -> fp16
    // In this case, the inferred type has become i8, and we have to set it back to fp16
    auto newConvShape = getShape(convOp.output());
    auto originOutType = origOp.output().getType().dyn_cast<vpux::NDTypeInterface>();
    auto newOtType = originOutType.changeShape(newConvShape);
    convOp.getResult().setType(newOtType);
    _log.trace("Insert new Convolution without batch: {0}", convOp);

    auto outTranspose = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, convOp.output(), nullptr, orderAttr);
    _log.trace("Insert transpose for output", outTranspose);

    return mlir::success();
}

//
// ConvertBatchedConvTo1NPass
//

class ConvertBatchedConvTo1NPass final : public IE::ConvertBatchedConvTo1NBase<ConvertBatchedConvTo1NPass> {
public:
    explicit ConvertBatchedConvTo1NPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertBatchedConvTo1NPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isPlaneEq1 = [](mlir::Value val) {
        const auto inputShape = getShape(val);
        const auto hCount = inputShape[Dims4D::Act::H];
        return hCount == 1;
    };

    const auto isBatchEq1 = [](mlir::Value val) {
        const auto inputShape = getShape(val);
        return inputShape[Dims4D::Act::N] == 1;
    };

    const auto shapeRankEq4 = [](mlir::Value val) {
        const auto inputShape = getShape(val);
        return inputShape.size() == 4;
    };

    const auto isPerAxisQuant = [](mlir::Value val) {
        auto elemType = val.getType().dyn_cast<vpux::NDTypeInterface>().getElementType();
        return elemType.isa<mlir::quant::UniformQuantizedPerAxisType>();
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) -> bool {
        auto hasPerAxisQuantization = isPerAxisQuant(op.input()) || isPerAxisQuant(op.output());
        auto isPlanesEq1 = isPlaneEq1(op.input()) && isPlaneEq1(op.filter());
        return !shapeRankEq4(op.input()) || isBatchEq1(op.input()) || !isPlanesEq1 || hasPerAxisQuantization;
    });
    target.addLegalOp<IE::TransposeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBatchedConvTo1NPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBatchedConvTo1NPass(Logger log) {
    return std::make_unique<ConvertBatchedConvTo1NPass>(log);
}
