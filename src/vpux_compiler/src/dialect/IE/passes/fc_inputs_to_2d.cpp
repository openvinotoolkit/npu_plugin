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
// FCInputsTo2dPass
//

class FCInputsTo2dPass final : public IE::FCInputsTo2dBase<FCInputsTo2dPass> {
public:
    explicit FCInputsTo2dPass(Logger log) {
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

class FCInputsTo2dPass::MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FCInputsTo2dPass::MatMulOpConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    const auto lhsShape = getShape(matmulOp.input1());
    const auto rhsShape = getShape(matmulOp.input2());

    if (lhsShape.size() <= 2 || rhsShape.size() <= 2) {
        return mlir::failure();
    }

    int64_t lhsBatch = 1;
    int64_t lhsWidth = 1;
    int64_t lhsHeight = 1;
    auto lhsChannelDim = Dim(0);
    if (lhsShape.size() == 3) {
        lhsBatch = lhsShape[Dim(0)];
        lhsHeight = lhsShape[Dim(1)];
        lhsWidth = lhsShape[Dim(2)];
        lhsChannelDim = Dim(0);
    } else if (lhsShape.size() == 4) {
        lhsBatch = lhsShape[Dim(1)];
        lhsHeight = lhsShape[Dim(2)];
        lhsWidth = lhsShape[Dim(3)];
        lhsChannelDim = Dim(1);
    } else {
        lhsBatch = 1;
    }
    SmallVector<mlir::Value> activationSlices;
    if (lhsBatch > 1) {
        for (int64_t sliceIdx = 0; sliceIdx < lhsBatch; sliceIdx++) {
            Shape lhsOffsets = Shape(lhsShape.size(), 0);
            lhsOffsets[lhsChannelDim] = checked_cast<int64_t>(sliceIdx);
            auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), lhsOffsets);

            Shape lhsSizes = lhsShape.raw();
            lhsSizes[lhsChannelDim] = 1;
            auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), lhsSizes);
            // FIXME rewriter.createOrFold??
            auto newSubViewOp = rewriter.create<IE::SliceOp>(matmulOp->getLoc(), matmulOp.input1(), staticOffsetsAttr,
                                                             staticSizesAttr);

            Shape lhsShape2D{lhsHeight, lhsWidth};
            const auto lhsShape2DAttr = getIntArrayAttr(rewriter.getContext(), lhsShape2D);
            auto lhs2d =
                    rewriter.create<IE::ReshapeOp>(matmulOp->getLoc(), newSubViewOp, nullptr, false, lhsShape2DAttr);
            activationSlices.push_back(lhs2d);
        }
    } else {
        activationSlices.push_back(matmulOp.input1());
    }

    int64_t rhsBatch = 1;
    int64_t rhsWidth = 1;
    int64_t rhsHeight = 1;
    auto rhsChannelDim = Dim(0);
    if (rhsShape.size() == 3) {
        rhsBatch = rhsShape[Dim(0)];
        rhsHeight = rhsShape[Dim(1)];
        rhsWidth = rhsShape[Dim(2)];
        rhsChannelDim = Dim(0);
    } else if (rhsShape.size() == 4) {
        rhsBatch = rhsShape[Dim(1)];
        rhsHeight = rhsShape[Dim(2)];
        rhsWidth = rhsShape[Dim(3)];
        rhsChannelDim = Dim(1);
    } else {
        rhsBatch = 1;
    }
    SmallVector<mlir::Value> weightSlices;
    if (rhsBatch > 1) {
        for (int64_t sliceIdx = 0; sliceIdx < rhsBatch; sliceIdx++) {
            Shape rhsOffsets = Shape(rhsShape.size(), 0);
            rhsOffsets[rhsChannelDim] = checked_cast<int64_t>(sliceIdx);
            auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), rhsOffsets);

            Shape rhsSizes = rhsShape.raw();
            rhsSizes[rhsChannelDim] = 1;
            auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), rhsSizes);
            auto newSubViewOp = rewriter.create<IE::SliceOp>(matmulOp->getLoc(), matmulOp.input2(), staticOffsetsAttr,
                                                             staticSizesAttr);

            Shape rhsShape2D{rhsHeight, rhsWidth};
            const auto rhsShape2DAttr = getIntArrayAttr(rewriter.getContext(), rhsShape2D);
            auto rhs2d =
                    rewriter.create<IE::ReshapeOp>(matmulOp->getLoc(), newSubViewOp, nullptr, false, rhsShape2DAttr);
            weightSlices.push_back(rhs2d);
        }
    } else {
        weightSlices.push_back(matmulOp.input2());
    }

    SmallVector<mlir::Value> matmulSlices;
    for (size_t sliceIdx = 0; sliceIdx < activationSlices.size() && sliceIdx < weightSlices.size(); sliceIdx++) {
        auto lhs2d = activationSlices[sliceIdx];
        auto rhs2d = weightSlices[sliceIdx];
        auto op = rewriter.create<IE::MatMulOp>(matmulOp->getLoc(), lhs2d, rhs2d, matmulOp.transpose_a(),
                                                matmulOp.transpose_b());
        matmulSlices.push_back(op.output());
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(matmulOp->getLoc(), matmulSlices, 0);

    const auto outShape4D = getShape(matmulOp.output());
    const auto outShape4DAttr = getIntArrayAttr(rewriter.getContext(), outShape4D);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, newConcat, nullptr, false, outShape4DAttr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void FCInputsTo2dPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::MatMulOp>([](IE::MatMulOp op) -> bool {
        // FIXME what to do when the shapes do not match?
        const auto input1Shape = getShape(op.input1());
        const auto input2Shape = getShape(op.input2());
        return input1Shape.size() <= 2 && input2Shape.size() <= 2;
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
// createFCInputsTo2dPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFCInputsTo2dPass(Logger log) {
    return std::make_unique<FCInputsTo2dPass>(log);
}
