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
    // EISW-31028
    const auto hasNot4DRank = [&](mlir::Value operand) {
        return operand.getType().cast<vpux::NDTypeInterface>().getRank() != QUANT_DEQUANT_RANK;
    };

    if (llvm::any_of(layer->getOperands(), hasNot4DRank)) {
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
    for (auto* user : origOp->getUsers()) {
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
    // origOp.dump();

    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());
    // if (layer->getNumOperands() != 1 || layer->getNumResults() != 1){
    //     return mlir::failure();
    // }

    auto inputs = layer.getInputs();
    vpux::IE::DequantizeOp dequantOp = nullptr;
    size_t dequantInd;

    for (size_t inputInd = 0; inputInd < inputs.size(); ++inputInd) {
        auto input = inputs[inputInd];

        if (input.getDefiningOp<Const::DeclareOp>() == nullptr) {
            dequantOp = input.getDefiningOp<IE::DequantizeOp>();
            dequantInd = inputInd;
            if (dequantOp == nullptr) {
                return mlir::failure();
            }
        }
    }

    if (dequantOp == nullptr) {
        return mlir::failure();
    }

    // Check that tensor rank is 4, otherwise compilation fails in later passes
    // EISW-31028
    const auto hasNot4DRank = [&](mlir::Type resultType) {
        return resultType.cast<vpux::NDTypeInterface>().getRank() != QUANT_DEQUANT_RANK;
    };

    if (llvm::any_of(layer->getResultTypes(), hasNot4DRank)) {
        return mlir::failure();
    }

    // Check if operation supports quantization params propagation.
    const auto quantizedElemType = dequantOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto elemTypeInfo = origOp.getElemTypeInfo();

    elemTypeInfo.setInput(dequantInd, quantizedElemType);
    origOp.inferElemTypeInfo(elemTypeInfo);

    if (elemTypeInfo.getInput(dequantInd) != quantizedElemType ||
        !elemTypeInfo.getOutput(0).isa<mlir::quant::QuantizedType>()) {
        return matchFailed(rewriter, origOp, "Operation does not support quantization params propagation");
    }

    // Rewrite the sub-graph.
    rewriter.startRootUpdate(origOp);
    origOp->getOpOperand(dequantInd).set(dequantOp.input());

    // for (size_t inputInd = 0; inputInd < inputs.size(); ++inputInd) {
    //     auto input = layer.getInputs()[inputInd].getDefiningOp();
    //     input->dump();
    // }

    for (size_t inputInd = 0; inputInd < inputs.size(); ++inputInd) {
        if (inputInd == dequantInd) {
            continue;
        }

        auto input = layer.getInputs()[inputInd];
        auto inputOp = input.getDefiningOp();

        if (input != nullptr && (input.getType() != quantizedElemType)) {
            const auto output = inputOp->getOpResult(0);
            rewriter.setInsertionPointAfter(inputOp);
            auto newQuant = rewriter.create<IE::QuantizeOp>(origOp->getLoc(), output, quantizedElemType);
            layer->getOpOperand(inputInd).set(newQuant.output());
        }
    }

    for (size_t inputInd = 0; inputInd < inputs.size(); ++inputInd) {
        auto input = layer.getInputs()[inputInd].getDefiningOp();
        input->dump();
        input->getOpResult(0);
    }

    // infer return type
    mlir::SmallVector<mlir::Type> inferredTypes;
    auto op = mlir::cast<mlir::InferTypeOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(op.inferReturnTypes(getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                          op->getRegions(), inferredTypes)
                              .succeeded(),
                      "New type inference failed for '{0}'", op);

    for (size_t outputInd = 0; outputInd < origOp->getResults().size(); ++outputInd) {
        origOp->getResult(outputInd).setType(inferredTypes[outputInd]);

        const auto output = origOp->getOpResult(outputInd);
        rewriter.setInsertionPointAfter(origOp);
        auto newDequant = rewriter.create<IE::DequantizeOp>(dequantOp.getLoc(), output, dequantOp.dstElemType());
        output.replaceAllUsesExcept(newDequant.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});
    }
    rewriter.finalizeRootUpdate(origOp);
    return mlir::success();
}  // namespace

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
