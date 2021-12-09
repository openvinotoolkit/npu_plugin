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

private:
    void safeRunOnFunc() final;
};

//
// OpConverter
//

template <class concreteOp>
class OpConverterBase : public mlir::OpRewritePattern<concreteOp> {
public:
    using concreteOpTy = concreteOp;
    OpConverterBase(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<concreteOpTy>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(concreteOpTy origOp, mlir::PatternRewriter& rewriter) const final;

    virtual mlir::LogicalResult sliceInputs(mlir::PatternRewriter&, mlir::OpState*, int64_t,
                                            SmallVector<mlir::Value>&) const = 0;
    virtual mlir::LogicalResult appendOperationsToSlices(mlir::PatternRewriter&, mlir::OpState*,
                                                         llvm::SmallVector<mlir::Value>, mlir::Value&) const = 0;

private:
    Logger _log;
};

template <class concreteOp, int inputN>
class OpConverter : public OpConverterBase<concreteOp> {
public:
    const size_t numSlices = checked_cast<size_t>(inputN);

    OpConverter(mlir::MLIRContext* ctx, Logger log): OpConverterBase<concreteOp>(ctx, log) {
    }

    mlir::LogicalResult sliceInputs(mlir::PatternRewriter& rewriter, mlir::OpState* origOp, int64_t sliceIdx,
                                    SmallVector<mlir::Value>& slices) const override {
        auto newOpPtr = mlir::dyn_cast_or_null<concreteOp>(origOp);
        if (newOpPtr == nullptr) {
            return mlir::failure();
        }
        auto newOp = *newOpPtr;
        for (const auto inputIdx : irange(numSlices)) {
            const auto operands = newOp.getOperands();
            VPUX_THROW_UNLESS(operands.size() >= numSlices, "Not enough operands to slice");
            const auto input = operands[inputIdx];
            const auto shape = getShape(input);
            Shape offsets = Shape(shape.size(), 0);
            offsets[Dims4D::Act::N] = checked_cast<int64_t>(sliceIdx);
            const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);

            Shape sizes = shape.raw();
            VPUX_THROW_UNLESS(Dims4D::Act::N.ind() < checked_cast<int32_t>(sizes.size()),
                              "The shape rank must be 4 or more");
            sizes[Dims4D::Act::N] = 1;
            const auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sizes);

            const auto subViewOp = rewriter.createOrFold<IE::SliceOp>(newOp->getLoc(), input, offsetsAttr, sizesAttr);
            slices.push_back(subViewOp);
        }
        return mlir::success();
    }

    mlir::LogicalResult appendOperationsToSlices(mlir::PatternRewriter&, mlir::OpState*, llvm::SmallVector<mlir::Value>,
                                                 mlir::Value&) const override {
        VPUX_THROW("Unimplemented method");
    }
};

template <class concreteOp>
class EltwiseOpConverter : public OpConverter<concreteOp, 2> {
public:
    EltwiseOpConverter(mlir::MLIRContext* ctx, Logger log): OpConverter<concreteOp, 2>(ctx, log) {
    }

    mlir::LogicalResult appendOperationsToSlices(mlir::PatternRewriter& rewriter, mlir::OpState* origOp,
                                                 llvm::SmallVector<mlir::Value> slices,
                                                 mlir::Value& output) const override {
        auto newOpPtr = mlir::dyn_cast_or_null<concreteOp>(origOp);
        if (newOpPtr == nullptr) {
            return mlir::failure();
        }
        auto newOp = *newOpPtr;
        if (this->numSlices != slices.size()) {
            return mlir::failure();
        }
        auto lhs = slices[0];
        auto rhs = slices[1];
        auto newType = changeShape(newOp.getType(), getShape(lhs));
        auto op = rewriter.create<concreteOp>(newOp->getLoc(), newType, lhs, rhs, newOp.auto_broadcastAttr(),
                                              newOp.post_opAttr());
        output = op.output();
        return mlir::success();
    }
};

class ConvolutionOpConverter : public OpConverter<IE::ConvolutionOp, 1> {
public:
    ConvolutionOpConverter(mlir::MLIRContext* ctx, Logger log): OpConverter<concreteOpTy, 1>(ctx, log) {
    }

    mlir::LogicalResult appendOperationsToSlices(mlir::PatternRewriter& rewriter, mlir::OpState* origOp,
                                                 llvm::SmallVector<mlir::Value> slices,
                                                 mlir::Value& output) const override {
        auto newOpPtr = mlir::dyn_cast_or_null<concreteOpTy>(origOp);
        if (newOpPtr == nullptr) {
            return mlir::failure();
        }
        auto newOp = *newOpPtr;
        if (numSlices != slices.size()) {
            return mlir::failure();
        }
        auto lhs = slices[0];
        auto op = rewriter.create<concreteOpTy>(newOp->getLoc(), lhs, newOp.filter(), newOp.bias(), newOp.strides(),
                                                newOp.pads_begin(), newOp.pads_end(), newOp.dilations(),
                                                newOp.post_opAttr());
        output = op.output();
        return mlir::success();
    }
};

class FullyConnectedOpConverter : public OpConverter<IE::FullyConnectedOp, 1> {
public:
    FullyConnectedOpConverter(mlir::MLIRContext* ctx, Logger log): OpConverter<concreteOpTy, 1>(ctx, log) {
    }

    mlir::LogicalResult appendOperationsToSlices(mlir::PatternRewriter& rewriter, mlir::OpState* origOp,
                                                 llvm::SmallVector<mlir::Value> slices,
                                                 mlir::Value& output) const override {
        auto newOpPtr = mlir::dyn_cast_or_null<concreteOpTy>(origOp);
        if (newOpPtr == nullptr) {
            return mlir::failure();
        }
        auto newOp = *newOpPtr;
        if (numSlices != slices.size()) {
            return mlir::failure();
        }
        auto lhs = slices[0];
        auto op = rewriter.create<concreteOpTy>(newOp->getLoc(), lhs, newOp.weights(), newOp.bias());
        output = op.output();
        return mlir::success();
    }
};

// The following code written under the assumption that we will unroll batch of first N operands with equal batch
// dimension
template <class concreteOp>
mlir::LogicalResult OpConverterBase<concreteOp>::matchAndRewrite(concreteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto operands = origOp.getOperands();
    VPUX_THROW_WHEN(operands.empty(), "No operands to slice");
    const auto input1 = operands[0];
    const auto input1Shape = getShape(input1);
    VPUX_THROW_UNLESS(Dims4D::Act::N.ind() < checked_cast<int32_t>(input1Shape.size()),
                      "The shape rank must be 4 or more");
    const auto rowCount = input1Shape[Dims4D::Act::N];
    SmallVector<mlir::Value> slicesToConcat;
    for (const auto sliceIdx : irange(rowCount)) {
        SmallVector<mlir::Value> slices;
        if (mlir::failed(sliceInputs(rewriter, &origOp, sliceIdx, slices))) {
            return mlir::failure();
        }
        mlir::Value output;
        if (mlir::failed(appendOperationsToSlices(rewriter, &origOp, slices, output))) {
            return mlir::failure();
        }
        slicesToConcat.push_back(output);
    }
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slicesToConcat, Dims4D::Act::N.ind());

    return mlir::success();
}

//
// safeRunOnFunc
//

void UnrollBatchPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isBatchEq1 = [](mlir::Value val) {
        const auto inputShape = getShape(val);
        VPUX_THROW_UNLESS(Dims4D::Act::N.ind() < checked_cast<int32_t>(inputShape.size()),
                          "The shape rank must be 4 or more");
        const auto rowCount = inputShape[Dims4D::Act::N];
        return rowCount == 1;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::FullyConnectedOp>([&](IE::FullyConnectedOp op) -> bool {
        return isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) -> bool {
        return isBatchEq1(op.input());
    });
    target.addDynamicallyLegalOp<IE::AndOp>([&](IE::AndOp op) -> bool {
        return isBatchEq1(op.input1()) || getShape(op.input1()) != getShape(op.input2());
    });
    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) -> bool {
        return isBatchEq1(op.input1()) || getShape(op.input1()) != getShape(op.input2());
    });
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FullyConnectedOpConverter>(&ctx, _log);
    patterns.insert<ConvolutionOpConverter>(&ctx, _log);
    patterns.insert<EltwiseOpConverter<IE::AndOp>>(&ctx, _log);
    patterns.insert<EltwiseOpConverter<IE::AddOp>>(&ctx, _log);

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
