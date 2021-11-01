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
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>

using namespace vpux;

namespace {

constexpr int64_t QUANT_DEQUANT_RANK = 4;

template <class ConcreteOp>
bool checkIfAblePropagateTypeDown(ConcreteOp, mlir::quant::QuantizedType) {
    return true;
}

template <class ConcreteOp>
mlir::FailureOr<mlir::quant::QuantizedType> checkIfAblePropagateTypeUp(ConcreteOp, mlir::quant::QuantizedType type) {
    return type;
}

// TODO It needs further investigation how to propagate Transpose through per axis quant
template <>
bool checkIfAblePropagateTypeDown<IE::TransposeOp>(IE::TransposeOp, mlir::quant::QuantizedType type) {
    return !type.isa<mlir::quant::UniformQuantizedPerAxisType>();
}

template <>
mlir::FailureOr<mlir::quant::QuantizedType> checkIfAblePropagateTypeUp<IE::TransposeOp>(
        IE::TransposeOp, mlir::quant::QuantizedType type) {
    if (type.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return mlir::failure();
    }

    return type;
}

mlir::FailureOr<mlir::quant::QuantizedType> checkShapePropagation(mlir::quant::QuantizedType type, ShapeRef prevShape,
                                                                  ShapeRef newShape) {
    if (const auto perAxisType = type.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), prevShape, newShape);

        if (!axis.hasValue()) {
            return mlir::failure();
        }

        const auto quantizedDimension = axis.getValue();
        if (quantizedDimension == perAxisType.getQuantizedDimension()) {
            return type;
        }

        const auto newQuantDimSize = checked_cast<size_t>(newShape[Dim(quantizedDimension)]);
        if (newQuantDimSize != perAxisType.getScales().size()) {
            return mlir::failure();
        }

        return changeAxis(perAxisType, quantizedDimension);
    }

    return type;
}

template <>
bool checkIfAblePropagateTypeDown<IE::AffineReshapeOp>(IE::AffineReshapeOp op, mlir::quant::QuantizedType type) {
    return mlir::succeeded(checkShapePropagation(type, getShape(op.input()), getShape(op.output())));
}

template <>
mlir::FailureOr<mlir::quant::QuantizedType> checkIfAblePropagateTypeUp<IE::AffineReshapeOp>(
        IE::AffineReshapeOp op, mlir::quant::QuantizedType type) {
    return checkShapePropagation(type, getShape(op.output()), getShape(op.input()));
}

template <>
bool checkIfAblePropagateTypeDown<IE::ReshapeOp>(IE::ReshapeOp op, mlir::quant::QuantizedType type) {
    return mlir::succeeded(checkShapePropagation(type, getShape(op.input()), getShape(op.output())));
}

template <>
mlir::FailureOr<mlir::quant::QuantizedType> checkIfAblePropagateTypeUp<IE::ReshapeOp>(IE::ReshapeOp op,
                                                                                      mlir::quant::QuantizedType type) {
    return checkShapePropagation(type, getShape(op.output()), getShape(op.input()));
}

//
// PropagateQuantize
//

template <class ConcreteOp>
class PropagateQuantize final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    PropagateQuantize(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("PropagateQuantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::FailureOr<mlir::quant::QuantizedType> getElementType(mlir::quant::QuantizedType type, ConcreteOp op) const;

    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult PropagateQuantize<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto sourceOp = quantizeOp.input().getDefiningOp<ConcreteOp>();
    if (sourceOp == nullptr) {
        return mlir::failure();
    }

    auto originDstType = quantizeOp.dstElemType().cast<mlir::quant::QuantizedType>();

    auto newDstType = checkIfAblePropagateTypeUp(sourceOp, originDstType);
    if (mlir::failed(newDstType)) {
        return mlir::failure();
    }

    const auto isSameQuantize = [&](mlir::Operation* user) {
        if (auto currentQuantize = mlir::dyn_cast<IE::QuantizeOp>(user)) {
            return currentQuantize.dstElemType() == quantizeOp.dstElemType();
        }

        return false;
    };

    if (!llvm::all_of(sourceOp->getUsers(), isSameQuantize)) {
        return mlir::failure();
    }

    const auto hasNot4DRank = [&](mlir::Value operand) {
        return operand.getType().cast<mlir::ShapedType>().getRank() != QUANT_DEQUANT_RANK;
    };

    if (llvm::any_of(sourceOp->getOperands(), hasNot4DRank)) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> operands;
    for (auto operand : sourceOp->getOperands()) {
        auto newQuantize = rewriter.create<IE::QuantizeOp>(quantizeOp->getLoc(), operand, newDstType.getValue());
        operands.push_back(newQuantize.output());
    }

    auto newOp = rewriter.create<ConcreteOp>(sourceOp->getLoc(), operands, sourceOp->getAttrs());

    for (auto* user : sourceOp->getUsers()) {
        rewriter.replaceOp(user, newOp->getResults());
    }

    return mlir::success();
}

//
// PropagateDequantize
//

template <class ConcreteOp>
class PropagateDequantize final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    PropagateDequantize(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
        this->setDebugName("PropagateDequantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp dequantOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult PropagateDequantize<ConcreteOp>::matchAndRewrite(IE::DequantizeOp dequantOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    const auto inType = dequantOp.input().getType().cast<mlir::ShapedType>();
    const auto inElemType = inType.getElementType().cast<mlir::quant::QuantizedType>();

    const auto validateOp = [&](mlir::Operation* user) {
        return mlir::isa<ConcreteOp>(user) &&
               user->getResult(0).getType().cast<mlir::ShapedType>().getRank() == QUANT_DEQUANT_RANK &&
               checkIfAblePropagateTypeDown(mlir::cast<ConcreteOp>(user), inElemType);
    };

    if (!llvm::all_of(dequantOp->getUsers(), validateOp)) {
        return mlir::failure();
    }

    for (auto* user : dequantOp->getUsers()) {
        auto newOp = rewriter.create<ConcreteOp>(user->getLoc(), dequantOp.input(), user->getAttrs());

        auto newDequant =
                rewriter.create<IE::DequantizeOp>(dequantOp.getLoc(), newOp.getResult(), dequantOp.dstElemType());

        rewriter.replaceOp(user, newDequant.output());
    }

    return mlir::success();
}

//
// PropagateQuantizeDequantizePass
//

class PropagateQuantizeDequantizePass final :
        public IE::PropagateQuantizeDequantizeBase<PropagateQuantizeDequantizePass> {
public:
    explicit PropagateQuantizeDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagateQuantizeDequantizePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.add<PropagateQuantize<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<PropagateDequantize<IE::AffineReshapeOp>>(&ctx, _log);
    patterns.add<PropagateQuantize<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<PropagateDequantize<IE::ReshapeOp>>(&ctx, _log);
    patterns.add<PropagateQuantize<IE::TransposeOp>>(&ctx, _log);
    patterns.add<PropagateDequantize<IE::TransposeOp>>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createPropagateQuantizeDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateQuantizeDequantizePass(Logger log) {
    return std::make_unique<PropagateQuantizeDequantizePass>(log);
}
