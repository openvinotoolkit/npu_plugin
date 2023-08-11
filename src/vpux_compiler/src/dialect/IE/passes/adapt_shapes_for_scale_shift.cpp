//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <vpux/compiler/utils/rewriter.hpp>

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/utils/strings.hpp"

using namespace vpux;

namespace {

//
// BroadcastEltwiseRewriter
//

template <typename EltwiseOp>
class BroadcastEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    BroadcastEltwiseRewriter<EltwiseOp>(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("BroadcastEltwiseRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(EltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult matchAndTranspose(EltwiseOp origOp, const int64_t broadcastAxis,
                                          mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

template <typename EltwiseOp>
mlir::LogicalResult BroadcastEltwiseRewriter<EltwiseOp>::matchAndTranspose(EltwiseOp origOp,
                                                                           const int64_t broadcastAxis,
                                                                           mlir::PatternRewriter& rewriter) const {
    // It is also possible to apply reshape to get `IE.Add : tensor<NxMx1x1>, tensor<1xMx1x1>`.
    // However, such approach may lead to a big cluster of NCE tasks after `UnrollBatch` pass.
    // The measurements show that transposition is more effective for this pass.
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());
    const auto ctx = origOp->getContext();
    const auto origOpLoc = origOp.getLoc();

    // Reshape NxM input to 1xNxMx1
    const auto lhsType = origOp.input1().getType().template cast<vpux::NDTypeInterface>();
    const auto lhsShape = lhsType.getShape();
    const std::array<int64_t, 4> lhs4D = {1, lhsShape[Dim(broadcastAxis - 1)], lhsShape[Dim(broadcastAxis)], 1};
    const auto reshapedLhsType = lhsType.changeShape(ShapeRef(lhs4D));
    const auto reshapedLhsLoc = appendLoc(origOpLoc, "reshape_lhs");
    auto reshapedLhs = rewriter.create<IE::ReshapeOp>(reshapedLhsLoc, reshapedLhsType, origOp.input1(), nullptr, false,
                                                      getIntArrayAttr(ctx, ShapeRef(lhs4D)));
    _log.trace("[{0}]: reshaped LHS: {1}", this->getDebugName(), reshapedLhsLoc);

    // Transpose 1xNxMx1 input 1xMxNx1
    const auto transposeLhsOrder =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 2, 1, 3}, ctx));
    const auto transposeLhsLoc = appendLoc(origOpLoc, "transpose_lhs");
    auto transposedLhs = rewriter.create<IE::TransposeOp>(transposeLhsLoc, reshapedLhs, nullptr, transposeLhsOrder);
    _log.trace("[{0}]: transposed LHS: {1}", this->getDebugName(), transposeLhsLoc);

    // Reshape 1xM input to 1xMx1x1
    const auto rhsType = origOp.input2().getType().template cast<vpux::NDTypeInterface>();
    const auto rhsShape = rhsType.getShape();
    const std::array<int64_t, 4> rhs4D = {1, rhsShape[Dim(broadcastAxis)], 1, 1};
    const auto reshapedRhsType = rhsType.changeShape(ShapeRef(rhs4D));
    const auto reshapedRhsLoc = appendLoc(origOpLoc, "reshape_rhs");
    auto reshapedRhs = rewriter.create<IE::ReshapeOp>(reshapedRhsLoc, reshapedRhsType, origOp.input2(), nullptr, false,
                                                      getIntArrayAttr(ctx, ShapeRef(rhs4D)));
    _log.trace("[{0}]: reshaped RHS: {1}", this->getDebugName(), reshapedRhs);

    // Create new IE.Add operation
    auto newOp = rewriter.create<EltwiseOp>(origOp->getLoc(), transposedLhs, reshapedRhs, origOp.auto_broadcast(),
                                            origOp.post_opAttr());
    _log.trace("[{0}]: new element-wise: {1}", this->getDebugName(), newOp);

    // Transpose the 1xMxNx1 output to 1xNxMx1
    const auto transposeOutOrder =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(SmallVector<uint32_t>{0, 2, 1, 3}, ctx));
    const auto transposeOutLoc = appendLoc(origOpLoc, "transpose_out");
    auto transposedOut = rewriter.create<IE::TransposeOp>(transposeOutLoc, newOp, nullptr, transposeOutOrder);
    _log.trace("[{0}]: transposed output: {1}", this->getDebugName(), transposedOut);

    // Reshape 1xNxMx1 output to NxM
    const auto reshapedOutType = origOp.output().getType().template cast<vpux::NDTypeInterface>();
    const auto outputShape = reshapedOutType.getShape();
    const auto reshapedOutLoc = appendLoc(origOpLoc, "reshape_out");
    auto reshapedOut = rewriter.create<IE::ReshapeOp>(reshapedOutLoc, reshapedOutType, transposedOut.output(), nullptr,
                                                      false, getIntArrayAttr(ctx, outputShape));

    _log.trace("[{0}]: reshaped output: {1}", this->getDebugName(), reshapedOut);

    rewriter.replaceOp(origOp, reshapedOut.output());

    return mlir::success();
}

template <typename EltwiseOp>
mlir::LogicalResult BroadcastEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    const auto outputShape = getShape(origOp->getResult(0));
    if (outputShape.size() == 2) {
        return matchAndTranspose(origOp, 1, rewriter);
    } else if (outputShape.size() == 3) {
        return matchAndTranspose(origOp, 2, rewriter);
    }

    return mlir::failure();
}

//
// AdaptShapesForScaleShiftPass
//

class AdaptShapesForScaleShiftPass final : public IE::AdaptShapesForScaleShiftPassBase<AdaptShapesForScaleShiftPass> {
public:
    explicit AdaptShapesForScaleShiftPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

// Examples of illegal operations:
// IE.Add : tensor<NxMxf16>, tensor<1xMxf16> -> tensor<NxMxf16>
// IE.Multiply : tensor<NxMxf16>, tensor<1xMxf16> -> tensor<NxMxf16>
// IE.Add : tensor<1xNxMxf16>, tensor<1x1xMxf16> -> tensor<1xNxMxf16>
// IE.Multiply : tensor<1xNxMxf16>, tensor<1x1xMxf16> -> tensor<1xNxMxf16>
bool isLegalEltwise(mlir::Operation* op) {
    if (op->getNumOperands() != 2) {
        return true;
    }
    const auto lhsInputShape = getShape(op->getOperand(0));
    const auto rhsInputShape = getShape(op->getOperand(1));
    // The pass is applicable only for 2d or 3d Add and Multiply
    const bool is2DCase = (lhsInputShape.size() == 2 && rhsInputShape.size() == 2);
    const bool is3DCase = (lhsInputShape.size() == 3 && rhsInputShape.size() == 3);
    if (!is2DCase && !is3DCase) {
        return true;
    }
    const auto nonTrivialDimPredicate = [](const int64_t dim) -> bool {
        return dim > 1;
    };
    // The pass is applicable only for Add and Multiply with 1xM or 1x1xM weight shapes
    const auto nonTrivialWeightDims =
            std::count_if(rhsInputShape.raw().begin(), rhsInputShape.raw().end(), nonTrivialDimPredicate);
    if (nonTrivialWeightDims != 1) {
        return true;
    }

    // Also, it doesn't make much sense to transpose pseudo-1d inputs like 1x1x512 or 1x512x1.
    // Therefore, IE.Add(1x1x512, 1x1x512) and IE.Add(1x512, 1x512) should be legal.
    const auto nonTrivialInputDims =
            std::count_if(lhsInputShape.raw().begin(), lhsInputShape.raw().end(), nonTrivialDimPredicate);
    if (nonTrivialInputDims <= 1) {
        return true;
    }

    // For 3d case, batch must be equal to 1
    if (is3DCase && (lhsInputShape[Dim(0)] != 1 || rhsInputShape[Dim(0)] != 1)) {
        return true;
    }
    const auto lhsInputWidth = lhsInputShape.raw().back();
    const auto rhsInputWidth = rhsInputShape.raw().back();
    // Check that input width is not 1 to avoid cases like IE.Add(1xNx1, 1xNx1).
    // Such operations already have broadcast over proper axis.
    return rhsInputWidth == 1 || lhsInputWidth != rhsInputWidth;
}

//
// TransposeEltwiseRewriter
//

template <typename EltwiseOp>
class TransposeEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    TransposeEltwiseRewriter<EltwiseOp>(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("TransposeEltwiseRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(EltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

SmallVector<Dim> getNonOneDim(ShapeRef inputShape) {
    SmallVector<Dim> nonOneDims;
    for (auto index : irange(inputShape.size())) {
        if (inputShape[Dim(index)] != 1) {
            nonOneDims.push_back(Dim(index));
        }
    }
    return nonOneDims;
}

SmallVector<mlir::Value> getActInputs(mlir::Operation* eltwiseOp) {
    SmallVector<mlir::Value> actInputs;
    llvm::copy_if(eltwiseOp->getOperands(), std::back_inserter(actInputs), [](mlir::Value operand) {
        return mlir::failed(IE::getConstParentOp(operand));
    });
    return actInputs;
}

SmallVector<mlir::Value> getConstInputs(mlir::Operation* eltwiseOp) {
    SmallVector<mlir::Value> constInputs;
    llvm::copy_if(eltwiseOp->getOperands(), std::back_inserter(constInputs), [](mlir::Value operand) {
        return mlir::succeeded(IE::getConstParentOp(operand));
    });
    return constInputs;
}

// Examples of illegal potential scaleshifts:
// IE.Multiply: tensor<NxCxHxWxf16>, tensor<1x1x1xWxf16> -> tensor<NxCxHxWxf16>
// IE.Subtract: tensor<NxCxHxWxf16>, tensor<1x1x1xWxf16> -> tensor<NxCxHxWxf16>
// IE.Add:      tensor<NxCxHxWxf16>, tensor<1x1x1xWxf16> -> tensor<NxCxHxWxf16>
bool isLegalScaleShift(mlir::Operation* op) {
    if (op->getNumOperands() != 2) {
        return true;
    }
    auto actInputs = getActInputs(op);
    auto constInputs = getConstInputs(op);
    if (actInputs.empty() || constInputs.empty()) {
        return true;
    }
    auto actInput = actInputs[0];
    auto constInput = constInputs[0];
    auto actInputShape = actInput.getType().cast<vpux::NDTypeInterface>().getShape();
    auto constInputShape = constInput.getType().cast<vpux::NDTypeInterface>().getShape();
    if (actInputShape.size() != 4 || constInputShape.size() != 4) {
        return true;
    }
    auto oneAndOnlyNonCDimNotOne = [&]() {
        SmallVector<Dim> nonOneDims = getNonOneDim(constInputShape);
        if (nonOneDims.size() != 1 || (nonOneDims[0] != Dims4D::Act::H && nonOneDims[0] != Dims4D::Act::W)) {
            return false;
        }
        if (constInputShape[nonOneDims[0]] != actInputShape[nonOneDims[0]]) {
            return false;
        }
        return true;
    };
    return !oneAndOnlyNonCDimNotOne();
}

template <typename EltwiseOp>
mlir::LogicalResult TransposeEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());
    const auto ctx = origOp->getContext();

    if (isLegalScaleShift(origOp.getOperation())) {
        return mlir::failure();
    }

    auto actInput = getActInputs(origOp.getOperation())[0];
    auto constInput = getConstInputs(origOp.getOperation())[0];
    auto actInputType = actInput.getType().template cast<vpux::NDTypeInterface>();
    auto actInputShape = actInputType.getShape();
    auto constInputType = constInput.getType().template cast<vpux::NDTypeInterface>();
    auto constInputShape = constInputType.getShape();
    auto outputType = origOp->getResult(0).getType().template cast<vpux::NDTypeInterface>();

    Dim dimToMove = *getNonOneDim(constInputShape).begin();
    auto newInputOrder =
            dimToMove == Dims4D::Act::W ? SmallVector<uint32_t>{0, 3, 1, 2} : SmallVector<uint32_t>{0, 2, 1, 3};
    auto newActInputShape = Shape({actInputShape[Dim(newInputOrder[0])], actInputShape[Dim(newInputOrder[1])],
                                   actInputShape[Dim(newInputOrder[2])], actInputShape[Dim(newInputOrder[3])]});
    auto newConstInputShape = Shape({constInputShape[Dim(newInputOrder[0])], constInputShape[Dim(newInputOrder[1])],
                                     constInputShape[Dim(newInputOrder[2])], constInputShape[Dim(newInputOrder[3])]});

    const auto transposedOrder = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(newInputOrder, ctx));

    auto actInputTranspose = rewriter.create<IE::TransposeOp>(
            origOp->getLoc(), actInputType.changeShape(newActInputShape), actInput, nullptr, transposedOrder);
    actInput.replaceUsesWithIf(actInputTranspose, [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == origOp;
    });
    auto constInputTranspose = rewriter.create<IE::TransposeOp>(
            origOp->getLoc(), constInputType.changeShape(newConstInputShape), constInput, nullptr, transposedOrder);
    constInput.replaceUsesWithIf(constInputTranspose, [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == origOp;
    });

    origOp->getResult(0).setType(outputType.changeShape(newActInputShape));

    // transpose back
    rewriter.setInsertionPointAfter(origOp);
    auto newOutputOrder =
            dimToMove == Dims4D::Act::W ? SmallVector<uint32_t>{0, 2, 3, 1} : SmallVector<uint32_t>{0, 2, 1, 3};
    const auto outputTransposeOrder = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(newOutputOrder, ctx));
    auto outputTranspose = rewriter.create<IE::TransposeOp>(origOp->getLoc(), outputType, origOp->getResult(0), nullptr,
                                                            outputTransposeOrder);
    origOp->getResult(0).replaceAllUsesExcept(outputTranspose, outputTranspose);
    return mlir::success();
}

void AdaptShapesForScaleShiftPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet transposePatterns(&ctx);
    transposePatterns.add<TransposeEltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);
    transposePatterns.add<TransposeEltwiseRewriter<IE::SubtractOp>>(&ctx, _log);
    transposePatterns.add<TransposeEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(transposePatterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }

    mlir::ConversionTarget broadcastTarget(ctx);
    broadcastTarget.addDynamicallyLegalOp<IE::AddOp>(isLegalEltwise);
    broadcastTarget.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalEltwise);
    broadcastTarget.addLegalOp<IE::TransposeOp, IE::ReshapeOp>();

    mlir::RewritePatternSet broadcastPatterns(&ctx);
    broadcastPatterns.add<BroadcastEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    broadcastPatterns.add<BroadcastEltwiseRewriter<IE::MultiplyOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, broadcastTarget, std::move(broadcastPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createAdaptShapesForScaleShiftPass(Logger log) {
    return std::make_unique<AdaptShapesForScaleShiftPass>(log);
}
