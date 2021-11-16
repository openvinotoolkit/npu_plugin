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
// UnrollBatchPass
//

class UnrollBatchPass final : public IE::UnrollBatchBase<UnrollBatchPass> {
public:
    explicit UnrollBatchPass(Logger log) {
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

class UnrollBatchPass::FullyConnectedOpConverter final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    FullyConnectedOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnrollBatchPass::FullyConnectedOpConverter::matchAndRewrite(IE::FullyConnectedOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    const auto input1Shape = getShape(origOp.input());
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

        rowSlices.push_back(newSubViewOp);
    }

    SmallVector<mlir::Value> fullyConnectedSlices;
    for (size_t sliceIdx = 0; sliceIdx < rowSlices.size(); sliceIdx++) {
        auto lhs2d = rowSlices[sliceIdx];
        auto rhs2d = origOp.weights();
        auto op = rewriter.create<IE::FullyConnectedOp>(origOp->getLoc(), lhs2d, rhs2d, origOp.bias());
        fullyConnectedSlices.push_back(op.output());
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, fullyConnectedSlices, batchDim.ind());

    return mlir::success();
}

//
// safeRunOnFunc
//

void UnrollBatchPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::FullyConnectedOp>([](IE::FullyConnectedOp op) -> bool {
        const auto inputShape = getShape(op.input());
        const auto rowCount = inputShape[Dim(0)];
        return rowCount == 1;
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FullyConnectedOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollBatchPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollBatchPass(Logger log) {
    return std::make_unique<UnrollBatchPass>(log);
}
