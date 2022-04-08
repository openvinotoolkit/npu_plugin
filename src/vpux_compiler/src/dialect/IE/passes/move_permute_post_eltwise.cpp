//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
using namespace IE;
namespace {

//
// MovePermutePostEltwisePass
//

class MovePermutePostEltwisePass final : public MovePermutePostEltwiseBase<MovePermutePostEltwisePass> {
public:
    MovePermutePostEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// PermuteEltwiseRewriter
//

template <class EltwiseOp>
class PermuteEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    PermuteEltwiseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<EltwiseOp>(ctx), _log(log) {
        this->setDebugName("PermuteEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Get the result Shape of the mapping from the source layout to the result layout
// with no data change
// e.g., source layout NCHW, shape is [n, c, h, w]
//       return layout NHWC, the mapped result shape should be [n, w, c, h]
Shape getMappedShape(DimsOrder sourceLayout, DimsOrder resultLayout, ShapeRef sourceShape) {
    auto resultShape = Shape(sourceShape.size(), 1);
    auto sourcePerm = sourceLayout.toPermutation();
    auto resultPerm = resultLayout.toPermutation();
    for (auto dim : irange(sourceShape.size())) {
        resultShape[resultPerm[dim]] = sourceShape[sourcePerm[dim]];
    }
    return resultShape;
}

mlir::FailureOr<IE::MemPermuteOp> getEltwiseInputPermute(const mlir::Value& eltwiseInput) {
    auto parentOp = eltwiseInput.getDefiningOp();
    while (parentOp) {
        if (auto parentPermute = mlir::dyn_cast<IE::MemPermuteOp>(parentOp)) {
            // Case when there is a MemPermute and the processing of EltWise above adds another MemPermute
            // Skipping now, as canonicalizer takes care of optimizing both MemPermutes
            auto grandParentOp = parentPermute.input().getDefiningOp();
            if (grandParentOp != nullptr && mlir::isa<IE::MemPermuteOp>(grandParentOp)) {
                auto grandParentPermute = mlir::cast<IE::MemPermuteOp>(grandParentOp);
                if (parentPermute.dst_order() == grandParentPermute.dst_order()) {
                    return mlir::failure();
                }
            }
            return parentPermute;
        } else if (auto parentQuantizeCast = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentQuantizeCast.getOperation())) {
                return mlir::failure();
            }
            parentOp = parentQuantizeCast.input().getDefiningOp();
            continue;
        } else {
            return mlir::failure();
        }
    }
    return mlir::failure();
}

/* Rewrite the pattern from:

   Permute      Permute
      |          |
(QuantizeCast) (QuantizeCast)
       \        /
         Eltwise
            |
      (QuantizeCast)
           ...

    to:
PermuteCast  PermuteCast
      |          |
 ShapeCast   ShapeCast
      |          |
(QuantizeCast) (QuantizeCast)
       \        /
         Eltwise
            |
     (QuantizeCast)
            |
        ShapeCast
            |
       PermuteCast
            |
         Permute
            |
           ...
 */
template <class EltwiseOp>
mlir::LogicalResult PermuteEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp eltwiseOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), eltwiseOp->getName(), eltwiseOp->getLoc());
    auto ctx = this->getContext();
    if (!eltwiseOp->template hasTrait<IE::EltwiseOp>()) {
        return mlir::failure();
    }
    auto permute1Result = getEltwiseInputPermute(eltwiseOp.input1());
    auto permute2Result = getEltwiseInputPermute(eltwiseOp.input2());
    auto bothInputsArePermutes = mlir::succeeded(permute1Result) && mlir::succeeded(permute2Result);
    if (!bothInputsArePermutes) {
        return mlir::failure();
    }
    auto permute1 = permute1Result.getValue();
    auto permute2 = permute2Result.getValue();
    auto permute1InputType = permute1.input().getType().template cast<vpux::NDTypeInterface>();
    auto permute2InputType = permute2.input().getType().template cast<vpux::NDTypeInterface>();
    auto permute1InputLayout = permute1InputType.getDimsOrder();
    auto permute2InputLayout = permute2InputType.getDimsOrder();
    auto eltwiseInput1Type = eltwiseOp.input1().getType().template cast<vpux::NDTypeInterface>();
    auto eltwiseInputLayout = eltwiseInput1Type.getDimsOrder();
    auto eltwiseInputShape = eltwiseInput1Type.getShape();
    auto eltwiseOutputLayout = eltwiseOp.output().getType().template cast<vpux::NDTypeInterface>().getDimsOrder();
    auto patternCanBeConverted = [&]() -> bool {
        if (permute1InputLayout != permute2InputLayout) {
            // If the two inputs before Permutes have different layouts, the case is not supported
            // e.g., input1 layout is NCHW, input2 layout is NWHC
            return false;
        }
        if (eltwiseInputLayout != DimsOrder::NHWC) {
            // Consider the layout is adjusted to NHWC for Eltwise
            return false;
        }
        if (eltwiseOutputLayout != eltwiseInputLayout) {
            return false;
        }
        return true;
    };
    if (!patternCanBeConverted()) {
        return mlir::failure();
    }
    _log.nest().trace("Moving permute op post eltwise {0} at {1}", eltwiseOp->getName(), eltwiseOp->getLoc());
    auto permutesToMove = (permute1 == permute2) ? SmallVector<IE::MemPermuteOp>({permute1})
                                                 : SmallVector<IE::MemPermuteOp>({permute1, permute2});
    const auto neutralMemPerm = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap({0, 1, 2, 3}, ctx));
    // Get operation skipping QuantizeCast ops
    auto getOwnerIgnoreQuantizeCast = [&](const mlir::OpOperand& opOperand) -> mlir::Operation* {
        auto ownerOp = opOperand.getOwner();
        while (ownerOp && mlir::isa<IE::QuantizeCastOp>(ownerOp) && !ownerOp->getResult(0).getUsers().empty()) {
            ownerOp = *ownerOp->getResult(0).getUsers().begin();
        }
        return ownerOp;
    };
    auto mappedShape = getMappedShape(permute1InputLayout, eltwiseInputLayout, eltwiseInputShape);

    for (auto curPermute : permutesToMove) {
        _log.nest().trace("Processing permute {0} {1}", curPermute->getName(), curPermute->getLoc());
        auto permuteOutputType = curPermute.output().getType().template cast<vpux::NDTypeInterface>();
        auto newPermuteCastOutputType = permuteOutputType.changeShape(mappedShape);
        const auto dstOrder = mlir::AffineMapAttr::get(eltwiseInputLayout.toAffineMap(ctx));
        rewriter.setInsertionPoint(curPermute);
        if (permute1InputLayout != eltwiseInputLayout) {
            auto permuteCast = rewriter.template create<IE::PermuteCastOp>(
                    curPermute->getLoc(), newPermuteCastOutputType, curPermute.input(), dstOrder, neutralMemPerm);
            auto shapeCast = rewriter.template create<IE::ShapeCastOp>(
                    curPermute->getLoc(), newPermuteCastOutputType.changeShape(eltwiseInputShape), permuteCast.output(),
                    getIntArrayAttr(ctx, eltwiseInputShape.raw()));
            curPermute.output().replaceUsesWithIf(shapeCast.result(), [&](mlir::OpOperand& opOperand) {
                return getOwnerIgnoreQuantizeCast(opOperand) == eltwiseOp;
            });
        } else {
            auto shapeCast = rewriter.template create<IE::ShapeCastOp>(
                    curPermute->getLoc(), newPermuteCastOutputType.changeShape(eltwiseInputShape), curPermute.input(),
                    getIntArrayAttr(ctx, eltwiseInputShape.raw()));
            curPermute.output().replaceUsesWithIf(shapeCast.result(), [&](mlir::OpOperand& opOperand) {
                return getOwnerIgnoreQuantizeCast(opOperand) == eltwiseOp;
            });
        }
    }

    // Get the output value of the last QuantizeCast op
    // or the output value of the Eltwise op if no QuantizeCast op exists
    auto getInsertPoint = [&]() -> mlir::Value {
        auto output = eltwiseOp.output();
        if (output.getUsers().empty()) {
            return output;
        }
        if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*output.getUsers().begin())) {
            return quantizeCastOp.output();
        }
        return output;
    };
    auto outputValue = getInsertPoint();
    auto eltwiseOutputType = outputValue.getType().template cast<vpux::NDTypeInterface>();
    rewriter.setInsertionPointAfter(outputValue.getDefiningOp());
    auto newOutputShapeCastType = eltwiseOutputType.changeShape(mappedShape);
    if (permute1InputLayout != eltwiseInputLayout) {
        auto outputShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType, outputValue, getIntArrayAttr(ctx, mappedShape.raw()));
        const auto outputDstOrder = mlir::AffineMapAttr::get(permute1InputLayout.toAffineMap(ctx));
        auto outputPermuteCast = rewriter.template create<IE::PermuteCastOp>(
                eltwiseOp->getLoc(), outputShapeCast.result(), outputDstOrder, neutralMemPerm);
        auto outputPermute = rewriter.template create<IE::MemPermuteOp>(
                eltwiseOp->getLoc(), outputPermuteCast.output(), permute1.dst_orderAttr(), permute1.mem_permAttr());
        outputPermute.output().setType(eltwiseOutputType);
        outputValue.replaceAllUsesExcept(outputPermute, outputShapeCast);
    } else {
        auto outputShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType.changeShape(permute1InputType.getShape()), outputValue,
                getIntArrayAttr(ctx, permute1InputType.getShape().raw()));
        auto outputPermute = rewriter.template create<IE::MemPermuteOp>(
                eltwiseOp->getLoc(), outputShapeCast.result(), permute1.dst_orderAttr(), permute1.mem_permAttr());
        outputPermute.output().setType(eltwiseOutputType);
        outputValue.replaceAllUsesExcept(outputPermute, outputShapeCast);
    }

    return mlir::success();
}

void MovePermutePostEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteEltwiseRewriter<IE::AddOp>>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createMovePermutePostEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMovePermutePostEltwisePass(Logger log) {
    return std::make_unique<MovePermutePostEltwisePass>(log);
}
