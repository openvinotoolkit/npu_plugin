//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

//
// PropagateQuantize
//

class PropagateQuantize final : public mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface> {
public:
    PropagateQuantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/* This rewriter searches for pattern:
fp_tensor -> [ElemTypeInfoOpInterface] -> fp_tensor -> [Quantize]        -> quantized_tensor
and replaces it with
fp_tensor -> [Quantize] -> quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor */
mlir::LogicalResult PropagateQuantize::matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());

    // 1. Get the first quantizeOp.
    auto quantizeOp = mlir::dyn_cast<IE::QuantizeOp>(*(layer->getUsers().begin()));
    if (quantizeOp == nullptr) {
        return mlir::failure();
    }

    // 2. Check that every user is Quantize op ant they are the same.
    const auto isSameQuantize = [&](mlir::Operation* user) {
        if (auto currentQuantize = mlir::dyn_cast<IE::QuantizeOp>(user)) {
            return currentQuantize.dstElemType() == quantizeOp.dstElemType();
        }

        return false;
    };

    if (!llvm::all_of(layer->getUsers(), isSameQuantize)) {
        return mlir::failure();
    }

    // 3. Check that tensor rank is 4, otherwise compilation fails in later passes
    // E#31028
    const auto hasNot4DRankNotElemType = [&](mlir::Value operand) {
        return operand.getType().cast<vpux::NDTypeInterface>().getRank() != QUANT_DEQUANT_RANK &&
               operand.getDefiningOp<IE::ElemTypeInfoOpInterface>() == nullptr;
    };

    if (llvm::any_of(layer->getOperands(), hasNot4DRankNotElemType)) {
        return mlir::failure();
    }

    // 4. Check that operation supports quantization params propagation.
    const auto quantizedElemType = quantizeOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto elemTypeInfo = origOp.getElemTypeInfo();

    elemTypeInfo.setOutput(0, quantizedElemType);
    origOp.inferElemTypeInfoUp(elemTypeInfo);

    if (elemTypeInfo.getOutput(0) != quantizedElemType || !elemTypeInfo.getInput(0).isa<mlir::quant::QuantizedType>()) {
        return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation");
    }

    // All checks passed. Rewrite the sub-graph.
    rewriter.startRootUpdate(origOp);
    rewriter.setInsertionPoint(origOp);

    // 1. Create new Quantize ops, place them on each input of current operation.
    for (auto& operand : origOp->getOpOperands()) {
        auto newQuantize =
                rewriter.create<IE::QuantizeOp>(quantizeOp->getLoc(), operand.get(), elemTypeInfo.getInput(0));
        // Update input of Operation. NewQuant -> current Op.
        operand.set(newQuantize.output());
    }

    // 2. Infer return types, set output type of operation to inferred quantized type.
    mlir::SmallVector<mlir::Type> inferredTypes;
    auto op = mlir::cast<mlir::InferTypeOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(
            op.inferReturnTypes(getContext(), op->getLoc(), origOp->getOperands(), op->getAttrDictionary(),  // operands
                                op->getRegions(), inferredTypes)
                    .succeeded(),
            "New type inference failed for '{0}'", op);
    for (auto result : origOp->getResults()) {
        result.setType(inferredTypes[0]);
    }

    // 3. remove old Quantize ops.
    for (auto* user : llvm::make_early_inc_range(origOp->getUsers())) {
        rewriter.replaceOp(user, origOp->getResults());
    }

    // Rewrite done.
    rewriter.finalizeRootUpdate(origOp);

    return mlir::success();
}

//
// PropagateDequantize
//

class PropagateDequantize final : public mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface> {
public:
    PropagateDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::ElemTypeInfoOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/* This rewriter searches for pattern:
quantized_tensor -> [Dequantize] -> fp_tensor -> [ElemTypeInfoOpInterface] 	                -> fp_tensor
and replaces it with
quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor(inferred) -> [Dequantize] -> fp_tensor */
mlir::LogicalResult PropagateDequantize::matchAndRewrite(IE::ElemTypeInfoOpInterface origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer: {0}", origOp);

    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());
    if (layer->getNumResults() != 1) {
        return mlir::failure();
    }

    // 1. Check if there is output below
    auto hasReturnConsumer = llvm::any_of(layer->getUsers(), [](auto user) {
        return mlir::isa<mlir::ReturnOp>(user);
    });
    if (hasReturnConsumer) {
        return matchFailed(rewriter, origOp, "Operation has Return op consumer");
    }

    // 2. All inputs are Dequantize ops with same destination element type
    SmallVector<IE::DequantizeOp> dequantizeOps;
    auto allInputsDequantize = llvm::all_of(layer.getInputs(), [&](mlir::Value input) {
        auto dequantizeOp = input.getDefiningOp<IE::DequantizeOp>();
        if (dequantizeOp == nullptr) {
            return false;
        }

        dequantizeOps.push_back(dequantizeOp);
        return true;
    });

    if (!allInputsDequantize) {
        return matchFailed(rewriter, origOp, "Not all inputs are Dequantize op");
    }

    auto firstDequantizeOp = dequantizeOps[0];
    auto differentDstElemType = llvm::any_of(drop_begin(dequantizeOps), [&](IE::DequantizeOp dequantizeOp) {
        return dequantizeOp.dstElemType() != firstDequantizeOp.dstElemType();
    });

    if (differentDstElemType) {
        return matchFailed(rewriter, origOp, "Dequantize inputs have different destination element type");
    }

    // 3. Check that tensor rank is 4, otherwise compilation fails in later passes
    // // E#31028
    const auto isElemType = [&](auto user) {
        return mlir::dyn_cast_or_null<IE::ElemTypeInfoOpInterface>(user) == nullptr;
    };

    if (layer->getResult(0).getType().cast<vpux::NDTypeInterface>().getRank() != QUANT_DEQUANT_RANK &&
        llvm::any_of(layer->getUsers(), isElemType)) {
        return mlir::failure();
    }

    // 4. Check if operation supports quantization params propagation.
    auto elemTypeInfo = origOp.getElemTypeInfo();

    SmallVector<mlir::Type> originalTypes;
    for (auto idx : irange(dequantizeOps.size())) {
        auto dequantizeOp = dequantizeOps[idx];

        const auto quantizedElemType = dequantizeOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
        elemTypeInfo.setInput(idx, quantizedElemType);
        originalTypes.push_back(quantizedElemType);
    }

    origOp.inferElemTypeInfo(elemTypeInfo);

    const auto typesAreOriginal = llvm::all_of(irange(originalTypes.size()), [&](size_t idx) {
        return elemTypeInfo.getInput(idx) == originalTypes[idx];
    });

    if (!typesAreOriginal || !elemTypeInfo.getOutput(0).isa<mlir::quant::QuantizedType>()) {
        return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation: {0}",
                           elemTypeInfo.getOutput(0));
    }

    // 5. Rewrite the sub-graph.
    rewriter.startRootUpdate(origOp);

    const auto inputs = origOp->getOpOperands();
    for (auto idx : irange(inputs.size())) {
        auto& input = inputs[idx];

        input.set(dequantizeOps[idx].input());
    }

    // infer return type
    mlir::SmallVector<mlir::Type> inferredTypes;
    auto op = mlir::cast<mlir::InferTypeOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(op.inferReturnTypes(getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                          op->getRegions(), inferredTypes)
                              .succeeded(),
                      "New type inference failed for '{0}'", op);
    origOp->getResult(0).setType(inferredTypes[0]);

    const auto output = origOp->getOpResult(0);
    rewriter.setInsertionPointAfter(origOp);
    auto newLoc = appendLoc(origOp->getLoc(), "_propagated_Dequantize");
    auto newDequant = rewriter.create<IE::DequantizeOp>(newLoc, output, firstDequantizeOp.dstElemType());
    _log.trace("Added new Dequantize op: {0}", newDequant);
    output.replaceAllUsesExcept(newDequant.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});
    _log.trace("All uses of current layer have been replaced with new Dequantize op");

    rewriter.finalizeRootUpdate(origOp);
    return mlir::success();
}

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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateQuantize>(&ctx, _log.nest());
    patterns.add<PropagateDequantize>(&ctx, _log.nest());

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
