//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
// UnrollBatchPass
//

class UnrollBatchPass final : public IE::UnrollBatchBase<UnrollBatchPass> {
public:
    explicit UnrollBatchPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// OpConverter
//

template <class ConcreteOp>
class OpConverter : public mlir::OpRewritePattern<ConcreteOp> {
public:
    OpConverter(mlir::MLIRContext* ctx, Logger log, size_t numInputs)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _numInputs(numInputs) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

    SmallVector<mlir::Value> sliceInputs(mlir::PatternRewriter& rewriter, ConcreteOp origOp, int64_t sliceIdx) const {
        const auto operands = origOp->getOperands();
        SmallVector<mlir::Value> slices;
        for (const auto inputIdx : irange(_numInputs)) {
            const auto input = operands[inputIdx];
            const auto prevOperands = operands.take_front(inputIdx);
            const auto similarInput = llvm::find(prevOperands, input);
            if (similarInput == prevOperands.end()) {
                const auto shape = getShape(input);
                Shape offsets = Shape(shape.size(), 0);
                offsets[Dim(0)] = checked_cast<int64_t>(sliceIdx);
                const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);

                Shape sizes = shape.raw();
                sizes[Dim(0)] = 1;
                const auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sizes);

                const auto subViewOp =
                        rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), input, offsetsAttr, sizesAttr);
                slices.push_back(subViewOp);
            } else {
                const auto similarSliceIdx = std::distance(prevOperands.begin(), similarInput);
                slices.push_back(slices[similarSliceIdx]);
            }
        }
        return slices;
    }

    mlir::Value appendOperationsToSlices(mlir::PatternRewriter& rewriter, ConcreteOp origOp,
                                         mlir::ValueRange slices) const {
        const auto origOperands = origOp->getOperands();

        mlir::BlockAndValueMapping mapper;
        mapper.map(origOperands.take_front(slices.size()), slices);

        auto* newOp = rewriter.clone(*origOp.getOperation(), mapper);
        inferReturnTypes(newOp, InferShapedTypeMode::SHAPE);

        return newOp->getResult(0);
    }

private:
    Logger _log;

protected:
    size_t _numInputs;
};

template <class ConcreteOp>
mlir::LogicalResult OpConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto operands = origOp->getOperands();
    VPUX_THROW_WHEN(operands.empty(), "No operands to slice");
    VPUX_THROW_WHEN(origOp->getNumResults() != 1, "Operations with multiple results are not supported");
    VPUX_THROW_UNLESS(operands.size() >= _numInputs,
                      "Not enough operands to slice. Not less than {0} expected, but {1} provided", _numInputs,
                      operands.size());

    const auto input1 = operands[0];
    const auto input1Shape = getShape(input1);
    const auto rowCount = input1Shape[Dim(0)];
    const auto operandsToSlice = operands.take_front(_numInputs);

    const bool isBatchEqual =
            std::all_of(operandsToSlice.begin(), operandsToSlice.end(), [rowCount](mlir::Value value) {
                return getShape(value)[Dim(0)] == rowCount;
            });
    VPUX_THROW_UNLESS(isBatchEqual, "The pass can only slice the inputs with equal batch dimension");

    SmallVector<mlir::Value> slicesToConcat;
    for (const auto sliceIdx : irange(rowCount)) {
        const auto slices = sliceInputs(rewriter, origOp, sliceIdx);
        VPUX_THROW_UNLESS(slices.size() == _numInputs, "Slices range must contain {0} values, but {1} provided",
                          _numInputs, slices.size());

        const auto output = appendOperationsToSlices(rewriter, origOp, slices);
        slicesToConcat.push_back(output);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slicesToConcat, Dim(0).ind());
    return mlir::success();
}

//
// safeRunOnFunc
//

void UnrollBatchPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isBatchEq1 = [](mlir::Value val) {
        const auto inputShape = getShape(val);
        const auto rowCount = inputShape[Dim(0)];
        return rowCount == 1;
    };

    const auto shapeRankEq0 = [](mlir::Value val) {
        const auto inputShape = getShape(val);
        return inputShape.size() == 0;
    };

    const auto isShapeRankEq = [](mlir::Value val1, mlir::Value val2) {
        const auto inputShape1 = getShape(val1);
        const auto inputShape2 = getShape(val2);
        return inputShape1.size() == inputShape2.size();
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::FullyConnectedOp>([&](IE::FullyConnectedOp op) -> bool {
        return shapeRankEq0(op.input()) || isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) -> bool {
        return shapeRankEq0(op.input()) || isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) -> bool {
        return shapeRankEq0(op.input()) || isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::ExpOp>([&](IE::ExpOp op) -> bool {
        return shapeRankEq0(op.input()) || isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::SigmoidOp>([&](IE::SigmoidOp op) -> bool {
        return shapeRankEq0(op.input()) || isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::AndOp>([&](IE::AndOp op) -> bool {
        return (shapeRankEq0(op.input1()) || shapeRankEq0(op.input2())) || !isShapeRankEq(op.input1(), op.input2()) ||
               (isBatchEq1(op.input1()) || isBatchEq1(op.input2()));
    });
    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) -> bool {
        return (shapeRankEq0(op.input1()) || shapeRankEq0(op.input2())) || !isShapeRankEq(op.input1(), op.input2()) ||
               (isBatchEq1(op.input1()) || isBatchEq1(op.input2()));
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OpConverter<IE::ConvolutionOp>>(&ctx, _log, 1);
    patterns.add<OpConverter<IE::FullyConnectedOp>>(&ctx, _log, 1);
    patterns.add<OpConverter<IE::GroupConvolutionOp>>(&ctx, _log, 1);
    patterns.add<OpConverter<IE::ExpOp>>(&ctx, _log, 1);
    patterns.add<OpConverter<IE::SigmoidOp>>(&ctx, _log, 1);
    patterns.add<OpConverter<IE::AndOp>>(&ctx, _log, 2);
    patterns.add<OpConverter<IE::AddOp>>(&ctx, _log, 2);

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
