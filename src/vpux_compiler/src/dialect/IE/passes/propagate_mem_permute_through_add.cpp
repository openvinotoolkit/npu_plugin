//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

mlir::Operation* getInputPermuteLikeOp(mlir::Value addInput) {
    auto parentOp = addInput.getDefiningOp();
    while (parentOp) {
        if (mlir::isa<IE::MemPermuteOp, IE::PermuteQuantizeOp>(parentOp)) {
            return parentOp;
        } else if (auto parentShapeCast = mlir::dyn_cast<IE::ShapeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentShapeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentShapeCast.source().getDefiningOp();
            continue;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

IE::AddOp getAddOp(mlir::Value permuteInput) {
    auto parentOp = permuteInput.getDefiningOp();
    while (parentOp) {
        if (auto parentAdd = mlir::dyn_cast<IE::AddOp>(parentOp)) {
            return parentAdd;
        } else if (auto parentQuantizeCast = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentQuantizeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentQuantizeCast.input().getDefiningOp();
            continue;
        } else if (auto parentShapeCast = mlir::dyn_cast<IE::ShapeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentShapeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentShapeCast.source().getDefiningOp();
            continue;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

//
// OptimizeEltwise
//

class OptimizeEltwise final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    OptimizeEltwise(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("OptimizeEltwise");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Propagate last permute in the chain IE.MemPermute -> IE.ShapeCast -> IE.Add -> IE.ShapeCast -> IE.MemPermute
// This subgraph becomes IE.MemPermute -> IE.MemPermute -> IE.ShapeCast -> IE.Add -> IE.ShapeCast
// Two consecutive IE.MemPermute operations will be folded into one.
// VPU.NCE.Eltwise is layout agnostic, however, DPU operates on NHWC layouts. Layout casts must be applied.
// IE.LayoutCast (to NCHW) -> IE.Add (NHWC input, NHWC output) -> IE.LayoutCast (to original)
mlir::LogicalResult OptimizeEltwise::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());
    auto ctx = memPermuteOp.getContext();
    auto quantizeCastOp = memPermuteOp.input().getDefiningOp<IE::QuantizeCastOp>();

    auto addOp = getAddOp(memPermuteOp.input());
    const SmallVector<mlir::Value> branches = addOp->getOperands();

    SmallVector<mlir::Value> newAddInputs;
    for (size_t inputIdx = 0; inputIdx < branches.size(); inputIdx++) {
        const auto inPermutationOp = getInputPermuteLikeOp(branches[inputIdx]);

        const auto newMemPermuteLoc = appendLoc(memPermuteOp.getLoc(), "_mem_permute_{0}", inputIdx);
        auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(newMemPermuteLoc, inPermutationOp->getResult(0),
                                                                 memPermuteOp.dst_order(), memPermuteOp.mem_perm());

        const auto addInShape = getShape(branches[inputIdx]).toValues();
        const auto addInShapeAttr = getIntArrayAttr(ctx, addInShape.raw());
        const auto origAddInType = branches[inputIdx].getType().cast<vpux::NDTypeInterface>();
        const auto newShapeCastOrder = DimsOrder::fromValue(newMemPermuteOp.output());
        const auto newShapeCastType = origAddInType.changeDimsOrder(newShapeCastOrder);
        auto newShapeCastOp =
                rewriter.create<IE::ShapeCastOp>(memPermuteOp.getLoc(), newShapeCastType.changeShape(addInShape),
                                                 newMemPermuteOp.output(), addInShapeAttr);

        const auto addInOrder = DimsOrder::fromValue(branches[inputIdx]);
        const auto orderInAttr = mlir::AffineMapAttr::get(addInOrder.toAffineMap(ctx));
        const auto inLayoutCastLoc = appendLoc(memPermuteOp.getLoc(), "_in_layout_cast_{0}", inputIdx);
        auto inLayoutCastOp = rewriter.create<IE::LayoutCastOp>(inLayoutCastLoc, newShapeCastOp.result(), orderInAttr);

        newAddInputs.push_back(inLayoutCastOp.output());
    }
    auto newAddOp = rewriter.create<IE::AddOp>(addOp.getLoc(), addOp.getType(), newAddInputs[0], newAddInputs[1],
                                               addOp.auto_broadcastAttr(), addOp.post_opAttr());

    const auto nceOutLayout = DimsOrder::fromValue(memPermuteOp.output());
    const auto orderOutAttr = mlir::AffineMapAttr::get(nceOutLayout.toAffineMap(ctx));
    const auto outLayoutCastLoc = appendLoc(memPermuteOp.getLoc(), "_out_layout_cast");
    auto outLayoutCastOp = rewriter.create<IE::LayoutCastOp>(outLayoutCastLoc, newAddOp.output(), orderOutAttr);

    const auto newOutShapeCastType = memPermuteOp.output().getType();
    const auto newOutShapeCastLoc = appendLoc(memPermuteOp.getLoc(), "_out_shape_cast");

    const Shape targetShape = getShape(memPermuteOp.output()).toValues();
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape.raw());
    IE::ShapeCastOp newOutShapeCastOp;
    if (quantizeCastOp != nullptr) {
        const auto quantizeCastInElemType = quantizeCastOp.input().getType().cast<NDTypeInterface>().getElementType();
        newOutShapeCastOp = rewriter.create<IE::ShapeCastOp>(
                newOutShapeCastLoc, newOutShapeCastType.cast<NDTypeInterface>().changeElemType(quantizeCastInElemType),
                outLayoutCastOp.output(), targetShapeAttr);
        auto newQuantizeCastOp = rewriter.create<IE::QuantizeCastOp>(
                quantizeCastOp.getLoc(), newOutShapeCastOp.result(), quantizeCastOp.dstElemTypeAttr());
        rewriter.replaceOp(memPermuteOp, newQuantizeCastOp.output());
    } else {
        newOutShapeCastOp = rewriter.create<IE::ShapeCastOp>(newOutShapeCastLoc, newOutShapeCastType,
                                                             outLayoutCastOp.output(), targetShapeAttr);
        rewriter.replaceOp(memPermuteOp, newOutShapeCastOp.result());
    }

    return mlir::success();
}

//
// PropagateMemPermuteThroughAddPass
//

class PropagateMemPermuteThroughAddPass final :
        public IE::PropagateMemPermuteThroughAddBase<PropagateMemPermuteThroughAddPass> {
public:
    explicit PropagateMemPermuteThroughAddPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool isSupportedMemPermute(IE::MemPermuteOp memPermuteOp, Logger log) const;

private:
    Logger _log;
};

// Search for pattern
// IE.MemPermute / PermuteQuantize -> [IE.ShapeCast]|
//                                                  | -> IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] -> IE.MemPermute
// IE.MemPermute / PermuteQuantize -> [IE.ShapeCast]|
bool canBeFolded(IE::PermuteQuantizeOp permuteQuantizeOp, IE::MemPermuteOp memPermuteOp) {
    const auto permuteQuantizeOutElemType =
            permuteQuantizeOp.output().getType().cast<vpux::NDTypeInterface>().getElementType();
    // Can fuse MemPermute with PermuteQuantization in case only permutation (no quantization) is performed by this
    // PermuteQuantization Op.
    if (permuteQuantizeOutElemType.isa<mlir::quant::QuantizedType>()) {
        return false;
    }

    auto prevMemPerm = permuteQuantizeOp.mem_perm();
    auto memPerm = memPermuteOp.mem_perm();
    auto newMemPerm = memPerm.compose(prevMemPerm);

    const auto permuteQuantizeOpInType = permuteQuantizeOp.input().getType();
    const auto memPermuteOpOutType = memPermuteOp.output().getType();
    auto permuteQuantizeOpInElemType = permuteQuantizeOpInType.cast<NDTypeInterface>().getElementType();
    // For the case that permutations can be folded, PermuteQuantizeOpInType and memPermuteOpOutType are expected to be
    // the same, except elemType.
    if (permuteQuantizeOpInType !=
                memPermuteOpOutType.cast<NDTypeInterface>().changeElemType(permuteQuantizeOpInElemType) ||
        !newMemPerm.isIdentity()) {
        return false;
    }

    return true;
}

bool canBeFusedIntoPermuteCast(IE::PermuteQuantizeOp permuteQuantizeOp, IE::MemPermuteOp memPermuteOp) {
    const auto inOrder = DimsOrder::fromValue(permuteQuantizeOp.input());
    const auto inShape = getShape(permuteQuantizeOp.input());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    auto prevMemPerm = permuteQuantizeOp.mem_perm();
    auto memPerm = memPermuteOp.mem_perm();
    auto composedMemPerm = memPerm.compose(prevMemPerm);

    if (isTrivialPermute(inMemShape, composedMemPerm)) {
        return true;
    }

    return false;
}

bool PropagateMemPermuteThroughAddPass::isSupportedMemPermute(IE::MemPermuteOp memPermuteOp, Logger log) const {
    auto addOp = getAddOp(memPermuteOp.input());
    if (addOp == nullptr) {
        log.trace("IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] -> IE.MemPermute pattern not found");
        return false;
    }

    const SmallVector<mlir::Value> branches = addOp->getOperands();
    for (const auto& addInput : branches) {
        const auto inPermutationOp = getInputPermuteLikeOp(addInput);
        if (inPermutationOp == nullptr) {
            log.trace("One of IE.Add inputs does not have IE.MemPermute or IE::PermuteQuantize");
            return false;
        }

        // Futher checking for inPermuteQuantizeOp - propagate if PermuteQuantize and MemPermute can be folded.
        auto inPermuteQuantizeOp = mlir::dyn_cast<IE::PermuteQuantizeOp>(inPermutationOp);
        if (inPermuteQuantizeOp != nullptr && !canBeFolded(inPermuteQuantizeOp, memPermuteOp) &&
            !canBeFusedIntoPermuteCast(inPermuteQuantizeOp, memPermuteOp)) {
            log.trace("IE::PermuteQuantize op: {0} and IE::MemPermute op: {1} can not be folded or fused into "
                      "permuteCast",
                      inPermuteQuantizeOp.getLoc(), memPermuteOp.getLoc());
            return false;
        }
    }

    log.trace("IE::MemPermute op: {0} can be converted", memPermuteOp.getLoc());
    return true;
}

void PropagateMemPermuteThroughAddPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeEltwise>(&ctx, _log);

    mlir::ConversionTarget target(ctx);
    const auto isLegalMemPermute = [&](IE::MemPermuteOp memPermuteOp) -> bool {
        return !isSupportedMemPermute(memPermuteOp, _log);
    };
    target.addDynamicallyLegalOp<IE::MemPermuteOp>(isLegalMemPermute);
    target.addLegalOp<IE::ShapeCastOp>();
    target.addLegalOp<IE::LayoutCastOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<IE::QuantizeCastOp>();

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateMemPermuteThroughAddPass(Logger log) {
    return std::make_unique<PropagateMemPermuteThroughAddPass>(log);
}
