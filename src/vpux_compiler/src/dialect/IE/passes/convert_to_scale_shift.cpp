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

using namespace vpux;

namespace {

mlir::LogicalResult verifyAndBroadcastInput(mlir::Location loc, mlir::Value& input, vpux::ShapeRef inputShape,
                                            vpux::ShapeRef outputShape, mlir::PatternRewriter& rewriter) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (outputShape.size() != 4 || inputShape.size() != 4) {
        return mlir::failure();
    }
    if (inputShape[N] != 1 || inputShape[H] != 1 || inputShape[W] != 1) {
        return mlir::failure();
    }

    if (inputShape[C] != outputShape[C] && inputShape[C] != 1) {
        return mlir::failure();
    }

    // Broadcast scalar for all channels
    if (inputShape[C] != outputShape[C] && inputShape[C] == 1) {
        auto input2Const = input.getDefiningOp<Const::DeclareOp>();
        if (input2Const == nullptr) {
            return mlir::failure();
        }
        Const::ContentAttr dataAttr = input2Const.contentAttr().broadcast(C, outputShape[C]);

        if (dataAttr == nullptr) {
            return mlir::failure();
        }

        auto dataConstOp = rewriter.create<Const::DeclareOp>(loc, dataAttr.getType(), dataAttr);

        input = dataConstOp.output();
    }

    return mlir::success();
}

//
// ConvertAddToScaleShift
//

class ConvertAddToScaleShift final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    ConvertAddToScaleShift(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        setDebugName("ConvertAddToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertAddToScaleShift::matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const {
    auto inElemType = biasOp.input2().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = biasOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    if (inElemType != outElemType) {
        return mlir::failure();
    }

    const auto lhsType = biasOp.input1().getType().cast<vpux::NDTypeInterface>();
    const auto outShapeRes = biasOp.output().getType().cast<vpux::NDTypeInterface>();

    bool lhsIsActivation = (lhsType == outShapeRes);
    auto activationInput = lhsIsActivation ? biasOp.input1() : biasOp.input2();
    auto biasInput = lhsIsActivation ? biasOp.input2() : biasOp.input1();

    auto mulOutShape = getShape(biasOp.output());
    auto biasesShape = getShape(biasInput);

    if (verifyAndBroadcastInput(biasOp.getLoc(), biasInput, biasesShape, mulOutShape, rewriter).failed())
        return mlir::failure();

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), activationInput, nullptr, biasInput);

    return mlir::success();
}

//
// ConvertMultiplyToScaleShift
//

class ConvertMultiplyToScaleShift final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    ConvertMultiplyToScaleShift(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        setDebugName("ConvertMultiplyToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertMultiplyToScaleShift::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto lhsType = mulOp.input1().getType().cast<mlir::ShapedType>();
    const auto outShapeRes = mulOp.output().getType().cast<mlir::ShapedType>();

    bool lhsIsActivation = (lhsType == outShapeRes);
    auto activationInput = lhsIsActivation ? mulOp.input1() : mulOp.input2();
    auto weightsInput = lhsIsActivation ? mulOp.input2() : mulOp.input1();

    auto mulOutShape = getShape(mulOp.output());
    auto weightsShape = getShape(weightsInput);

    if (verifyAndBroadcastInput(mulOp.getLoc(), weightsInput, weightsShape, mulOutShape, rewriter).failed())
        return mlir::failure();

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(mulOp, mulOp.getType(), activationInput, weightsInput, nullptr);
    return mlir::success();
}

//
// ConvertToScaleShiftPass
//

class ConvertToScaleShiftPass final : public IE::ConvertToScaleShiftBase<ConvertToScaleShiftPass> {
public:
    explicit ConvertToScaleShiftPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertToScaleShiftPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertAddToScaleShift>(&ctx, _log);
    patterns.insert<ConvertMultiplyToScaleShift>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToScaleShiftPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToScaleShiftPass(Logger log) {
    return std::make_unique<ConvertToScaleShiftPass>(log);
}
