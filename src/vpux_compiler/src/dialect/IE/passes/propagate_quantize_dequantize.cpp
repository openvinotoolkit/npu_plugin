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
// PropagateQuantizeWithSoftmax
//

class PropagateQuantizeWithSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    PropagateQuantizeWithSoftmax(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/* This rewriter searches for pattern:
fp_tensor -> [SoftMax] -> Quantize -> quantized_tensor
and replaces it with
fp_tensor -> Quantize -> quantized_tensor -> Dequantize -> fp_tensor -> [SoftMax] -> Quantize -> quantized_tensor */
mlir::LogicalResult PropagateQuantizeWithSoftmax::matchAndRewrite(IE::SoftMaxOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (origOp.input().getDefiningOp<IE::DequantizeOp>() != nullptr) {
        return mlir::failure();
    }

    auto outQuantizeOp = mlir::dyn_cast<IE::QuantizeOp>(*origOp->getUsers().begin());
    if (outQuantizeOp == nullptr) {
        return mlir::failure();
    }

    auto quantizeDstTypeAttr = outQuantizeOp.dstElemTypeAttr();

    auto newQuantizeOp = rewriter.create<IE::QuantizeOp>(origOp.getLoc(), origOp.input(), quantizeDstTypeAttr);
    auto softmaxInputType = origOp.input().getType().cast<vpux::NDTypeInterface>().getElementType();
    auto dequantizeOp = rewriter.create<IE::DequantizeOp>(origOp.getLoc(), newQuantizeOp.output(), softmaxInputType);
    rewriter.replaceOpWithNewOp<IE::SoftMaxOp>(origOp, dequantizeOp.output(), origOp.axisIndAttr());

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
    auto layer = mlir::cast<IE::LayerOpInterface>(origOp.getOperation());
    if (layer->getNumOperands() != 1 || layer->getNumResults() != 1) {
        return mlir::failure();
    }

    auto dequantOp = layer.getInputs()[0].getDefiningOp<IE::DequantizeOp>();
    if (dequantOp == nullptr) {
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
    patterns.insert<PropagateQuantizeWithSoftmax>(&ctx, _log.nest());
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
