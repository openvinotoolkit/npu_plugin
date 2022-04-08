//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// MatMulInputsTo2dPass
//

class MatMulInputsTo2dPass final : public IE::MatMulInputsTo2dBase<MatMulInputsTo2dPass> {
public:
    explicit MatMulInputsTo2dPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class MatMulOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// MatMulOpConverter
//

class MatMulInputsTo2dPass::MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
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
    SmallVector<mlir::Value> activationSlices = sliceTensor(matmulOp.input1(), matmulOp->getLoc(), rewriter);
    SmallVector<mlir::Value> weightSlices = sliceTensor(matmulOp.input2(), matmulOp->getLoc(), rewriter);

    SmallVector<mlir::Value> matmulSlices;
    for (size_t sliceIdx = 0; sliceIdx < activationSlices.size() && sliceIdx < weightSlices.size(); sliceIdx++) {
        auto lhs2d = activationSlices[sliceIdx];
        auto rhs2d = weightSlices[sliceIdx];
        auto op = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), lhs2d, rhs2d, matmulOp.transpose_a(),
                                                matmulOp.transpose_b());
        matmulSlices.push_back(op.output());
    }

    VPUX_THROW_WHEN(matmulSlices.empty(), "Cannot slice MatMul operation with input shape {0}, weights' shape {1}",
                    getShape(matmulOp.input1()), getShape(matmulOp.input2()));

    auto newOp = matmulSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(matmulOp->getLoc(), matmulSlices, 0)
                                          : matmulSlices.front();

    const auto outShape4D = getShape(matmulOp.output());
    const auto outShape4DAttr = getIntArrayAttr(rewriter.getContext(), outShape4D);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newOp, nullptr, false, outShape4DAttr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void MatMulInputsTo2dPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MatMulOp>([](IE::MatMulOp op) -> bool {
        const auto input1Shape = getShape(op.input1());
        const auto input2Shape = getShape(op.input2());
        // Cover 3D input or weights.
        if (input1Shape.size() == 3 && (input2Shape.size() == 3 || input2Shape.size() == 2)) {
            return false;
        }
        // Cover 4D input and weights without batch.
        if (input1Shape.size() == 4 && input2Shape.size() == 4 && input1Shape[Dim(0)] == 1) {
            return false;
        }
        // Other cases are not required at this point, therefore they're legal.
        return true;
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MatMulOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
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
