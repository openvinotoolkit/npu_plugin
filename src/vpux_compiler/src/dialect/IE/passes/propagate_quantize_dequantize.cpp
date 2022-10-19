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

    // 0. Check if there is input above
    if (llvm::any_of(layer->getOperands(), [](mlir::Value operand) {
            return operand.isa<mlir::BlockArgument>();
        })) {
        return mlir::failure();
    }

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

bool checkPropagationChain(IE::LayerOpInterface layer) {
    mlir::Operation* operation = layer;
    while (operation && !operation->getUsers().empty()) {
        auto user = *(operation->getUsers().begin());

        // TODO delete layers from the list as soon as it's moved to this pass
        // for now MultiplyOp is not converted to ScaleShift
        if (mlir::isa<IE::AlignedChannelsOpInterface, IE::ConcatOp, IE::SliceOp, IE::ConvertOp>(user) &&
            !mlir::isa<IE::MultiplyOp>(user)) {
            return true;
        }

        if (!mlir::isa<IE::ElemTypeInfoOpInterface>(user)) {
            return false;
        }

        operation = user;
    }

    return false;
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
    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());
    if (layer->getNumOperands() != 1 || layer->getNumResults() != 1) {
        return mlir::failure();
    }

    // 0. Check if there is output below
    if (llvm::any_of(layer->getUsers(), [](auto user) {
            return mlir::isa<mlir::ReturnOp>(user);
        })) {
        return mlir::failure();
    }

    auto dequantOp = layer.getInputs()[0].getDefiningOp<IE::DequantizeOp>();
    if (dequantOp == nullptr) {
        return mlir::failure();
    }

    // Check that tensor rank is 4, otherwise compilation fails in later passes
    // // E#31028
    const auto isElemType = [&](auto user) {
        return mlir::dyn_cast_or_null<IE::ElemTypeInfoOpInterface>(user) == nullptr;
    };

    if (layer->getResult(0).getType().cast<vpux::NDTypeInterface>().getRank() != QUANT_DEQUANT_RANK &&
        llvm::any_of(layer->getUsers(), isElemType)) {
        return mlir::failure();
    }

    // Check the chain of propagation
    // in case at the end of the chain we don't have dpu task or the layer
    // which might be quantized in the following pass - skip propagation
    if (!checkPropagationChain(layer)) {
        return mlir::failure();
    }

    // Check if operation supports quantization params propagation.
    const auto quantizedElemType = dequantOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto elemTypeInfo = origOp.getElemTypeInfo();

    elemTypeInfo.setInput(0, quantizedElemType);
    origOp.inferElemTypeInfo(elemTypeInfo);

    if (elemTypeInfo.getInput(0) != quantizedElemType || !elemTypeInfo.getOutput(0).isa<mlir::quant::QuantizedType>()) {
        return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation");
    }

    // Rewrite the sub-graph.
    rewriter.startRootUpdate(origOp);
    origOp->getOpOperand(0).set(dequantOp.input());

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
    auto newDequant = rewriter.create<IE::DequantizeOp>(dequantOp.getLoc(), output, dequantOp.dstElemType());
    output.replaceAllUsesExcept(newDequant.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});

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
    patterns.insert<PropagateQuantize>(&ctx, _log.nest());
    patterns.insert<PropagateDequantize>(&ctx, _log.nest());

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
