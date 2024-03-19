//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

//
// MatMulInputsTo2dPass
//

class MatMulInputsTo2dPass final : public IE::MatMulInputsTo2dBase<MatMulInputsTo2dPass> {
public:
    explicit MatMulInputsTo2dPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ReshapeNDInputConverter;
    class MatMulOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// MatMulOpConverter
//

class MatMulInputsTo2dPass::MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

static SmallVector<mlir::Value> sliceTensor(const mlir::Value tensorToSplit, const mlir::Location location,
                                            mlir::PatternRewriter& rewriter) {
    const auto tensorShape = getShape(tensorToSplit);
    int64_t batch = 1;
    int64_t width = 1;
    int64_t height = 1;
    auto channelDim = Dim(0);
    if (tensorShape.size() == 3) {
        batch = tensorShape[Dim(0)];
        height = tensorShape[Dim(1)];
        width = tensorShape[Dim(2)];
        channelDim = Dim(0);
    } else if (tensorShape.size() == 4) {
        batch = tensorShape[Dim(1)];
        height = tensorShape[Dim(2)];
        width = tensorShape[Dim(3)];
        channelDim = Dim(1);
    } else if (tensorShape.size() == 2) {
        return {tensorToSplit};
    }
    SmallVector<mlir::Value> weightSlices;
    Shape rhsShape2D{height, width};
    const auto rhsShape2DAttr = getIntArrayAttr(rewriter.getContext(), rhsShape2D);
    if (batch > 1) {
        for (int64_t sliceIdx = 0; sliceIdx < batch; sliceIdx++) {
            Shape sliceOffsets = Shape(tensorShape.size(), 0);
            sliceOffsets[channelDim] = checked_cast<int64_t>(sliceIdx);
            auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffsets);

            Shape sliceSizes = tensorShape.raw();
            sliceSizes[channelDim] = 1;
            auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSizes);
            auto newSubViewOp =
                    rewriter.create<IE::SliceOp>(location, tensorToSplit, staticOffsetsAttr, staticSizesAttr);

            auto rhs2d = rewriter.create<IE::ReshapeOp>(location, newSubViewOp, nullptr, false, rhsShape2DAttr);
            weightSlices.push_back(rhs2d);
        }
    } else {
        auto rhs2d = rewriter.create<IE::ReshapeOp>(location, tensorToSplit, nullptr, false, rhsShape2DAttr);
        weightSlices.push_back(rhs2d);
    }

    return weightSlices;
}

mlir::LogicalResult MatMulInputsTo2dPass::MatMulOpConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    // 1. Cover 3D input or weights.
    // 2. Cover 4D input and weights without batch.
    if (!(input1Shape.size() == 3 && input2Shape.size() == 3) &&
        !(input1Shape.size() == 4 &&
          ((input2Shape.size() == 4 || input2Shape.size() == 3) && input1Shape[Dim(0)] == 1))) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> activationSlices = sliceTensor(matmulOp.getInput1(), matmulOp->getLoc(), rewriter);
    SmallVector<mlir::Value> weightSlices = sliceTensor(matmulOp.getInput2(), matmulOp->getLoc(), rewriter);

    SmallVector<mlir::Value> matmulSlices;
    VPUX_THROW_UNLESS(activationSlices.size() == weightSlices.size() || weightSlices.size() == 1,
                      "Mismatch activationSlices number '{0}' with weightSlices number '{1}'", activationSlices.size(),
                      weightSlices.size());
    for (size_t sliceIdx = 0; sliceIdx < activationSlices.size(); sliceIdx++) {
        auto lhs2d = activationSlices[sliceIdx];
        auto rhs2d = weightSlices[weightSlices.size() == 1 ? 0 : sliceIdx];
        auto op = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), lhs2d, rhs2d, matmulOp.getTransposeA(),
                                                matmulOp.getTransposeB());
        matmulSlices.push_back(op.getOutput());
    }

    VPUX_THROW_WHEN(matmulSlices.empty(), "Cannot slice MatMul operation with input shape {0}, weights' shape {1}",
                    input1Shape, input2Shape);

    auto newOp = matmulSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(matmulOp->getLoc(), matmulSlices, 0)
                                          : matmulSlices.front();

    const auto outShape4D = getShape(matmulOp.getOutput());
    const auto outShape4DAttr = getIntArrayAttr(rewriter.getContext(), outShape4D);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newOp, nullptr, false, outShape4DAttr);

    return mlir::success();
}

//
// ReshapeNDInputConverter
//

class MatMulInputsTo2dPass::ReshapeNDInputConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    ReshapeNDInputConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MatMulInputsTo2dPass::ReshapeNDInputConverter::matchAndRewrite(
        IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), matmulOp->getName(), matmulOp->getLoc());

    auto transposeA = matmulOp.getTransposeA();
    auto transposeB = matmulOp.getTransposeB();
    auto input1Shape = getShape(matmulOp.getInput1());
    auto input2Shape = getShape(matmulOp.getInput2());

    if ((input1Shape.size() <= 3 && input2Shape.size() <= 3) && (input1Shape.size() == input2Shape.size())) {
        return mlir::failure();
    }

    if ((input1Shape.size() == 2 && !transposeA) || (input1Shape.size() == 3 && transposeA)) {
        return mlir::failure();
    }

    if (input2Shape.size() != 2 && input2Shape[Dim(0)] == 1) {
        return mlir::failure();
    }

    auto getLegalInputShape = [&](llvm::ArrayRef<int64_t> shape, int64_t reservedDim) {
        auto firstDim = std::accumulate(shape.begin(), shape.end() - reservedDim, 1, std::multiplies<int64_t>());
        auto newShape = Shape{firstDim};
        for (auto dimInd : irange(reservedDim)) {
            auto dim = shape.size() - reservedDim + dimInd;
            newShape.push_back(shape[dim]);
        }
        return newShape;
    };

    int64_t reservedDim = transposeA || (input2Shape.size() > 2)
                                  ? 2
                                  : 1;  // MatMul(2x3x4x5, 4x8) {transposeA = true} collapses to
                                        // MatMul(6x4x5, 4x8) {transposeA = true}
                                        // but then we don't need the transposition, we can collapse further
                                        // MatMul(2x3x4x5, 5x8) to MatMul(24x5, 5x8)

    Shape newIn1Shape = getLegalInputShape(input1Shape.raw(), reservedDim);
    auto reshapeInput1 = rewriter.createOrFold<IE::ReshapeOp>(matmulOp->getLoc(), matmulOp.getInput1(), nullptr, false,
                                                              getIntArrayAttr(rewriter.getContext(), newIn1Shape));

    auto shapeInput2 = matmulOp.getInput2();
    if (input2Shape.size() != 2) {
        int64_t reservedDim = 2;
        Shape newIn2Shape = getLegalInputShape(input2Shape.raw(), reservedDim);
        shapeInput2 = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                rewriter.createOrFold<IE::ReshapeOp>(matmulOp->getLoc(), matmulOp.getInput2(), nullptr, false,
                                                     getIntArrayAttr(rewriter.getContext(), newIn2Shape)));
    }
    auto newMatMul =
            rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), reshapeInput1, shapeInput2, transposeA, transposeB);

    const auto origOutShape = getShape(matmulOp.getOutput());
    const auto origOutShapeAttr = getIntArrayAttr(rewriter.getContext(), origOutShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newMatMul.getOutput(), nullptr, false, origOutShapeAttr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void MatMulInputsTo2dPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReshapeNDInputConverter>(&ctx, benefitLevels[0], _log);
    patterns.add<MatMulOpConverter>(&ctx, benefitLevels[1], _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMatMulInputsTo2dPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMatMulInputsTo2dPass(Logger log) {
    return std::make_unique<MatMulInputsTo2dPass>(log);
}
