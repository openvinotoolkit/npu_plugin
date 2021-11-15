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
// SplitFCInputByRowsPass
//

class SplitFCInputByRowsPass final : public IE::SplitFCInputByRowsBase<SplitFCInputByRowsPass> {
public:
    explicit SplitFCInputByRowsPass(Logger log) {
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

class SplitFCInputByRowsPass::FullyConnectedOpConverter final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    FullyConnectedOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitFCInputByRowsPass::FullyConnectedOpConverter::matchAndRewrite(
        IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto input1Shape = getShape(origOp.input());
    if (input1Shape.size() < 1) {
        return mlir::failure();
    }

    const auto batchDim = Dim(0);
    const auto rowCount = input1Shape[batchDim];
    if (rowCount <= 1) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> rowSlices;
    for (int64_t sliceIdx = 0; sliceIdx < rowCount; sliceIdx++) {
        Shape lhsOffsets = Shape(input1Shape.size(), 0);
        lhsOffsets[batchDim] = checked_cast<int64_t>(sliceIdx);
        auto staticOffsetsAttr = getIntArrayAttr(rewriter.getContext(), lhsOffsets);

        Shape lhsSizes = input1Shape.raw();
        lhsSizes[batchDim] = 1;
        auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), lhsSizes);
        auto newSubViewOp = rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), origOp.input(), staticOffsetsAttr,
                                                               staticSizesAttr);

        const int64_t rowWidth = input1Shape[Dim(1)];
        Shape rowShape{1, rowWidth};
        const auto rowShapeAttr = getIntArrayAttr(rewriter.getContext(), rowShape);
        auto row2d = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), newSubViewOp, nullptr, false, rowShapeAttr);
        rowSlices.push_back(row2d);
    }

    SmallVector<mlir::Value> fullyConnectedSlices;
    for (size_t sliceIdx = 0; sliceIdx < rowSlices.size(); sliceIdx++) {
        auto lhs2d = rowSlices[sliceIdx];
        auto rhs2d = origOp.weights();
        auto op = rewriter.create<IE::FullyConnectedOp>(origOp->getLoc(), lhs2d, rhs2d, origOp.bias());
        fullyConnectedSlices.push_back(op.output());
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), fullyConnectedSlices, 0);

    const auto outShape = getShape(origOp.output());
    const auto outShapeAttr = getIntArrayAttr(rewriter.getContext(), outShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newConcat, nullptr, false, outShapeAttr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitFCInputByRowsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::FullyConnectedOp>([](IE::FullyConnectedOp op) -> bool {
        const auto inputShape = getShape(op.input());
        if (inputShape.size() < 1) {
            // No idea what to do with such shape. Let it be legal.
            return true;
        }
        const auto rowCount = inputShape[Dim(0)];
        return rowCount == 1;
    });
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FullyConnectedOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitFCInputByRowsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitFCInputByRowsPass(Logger log) {
    return std::make_unique<SplitFCInputByRowsPass>(log);
}
